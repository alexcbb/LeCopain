import logging
import os
import time
from dataclasses import asdict, dataclass
from pprint import pformat

import rerun as rr

# from safetensors.torch import load_file, save_file
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from lerobot.common.policies.factory import make_policy
from lerobot.common.robot_devices.control_configs import (
    ControlConfig,
    RecordControlConfig,
    RemoteRobotConfig,
    TeleoperateControlConfig,
    ControlPipelineConfig,
)
from lerobot.common.robot_devices.control_utils import (
    control_loop,
    init_keyboard_listener,
    is_headless,
    reset_environment,
    sanity_check_dataset_name,
    stop_recording,
    warmup_record,
    predict_action,
    log_control_info
)
from lerobot.common.robot_devices.robots.configs import (
    FeetechMotorsBusConfig,
    OpenCVCameraConfig,
    So100RobotConfig,
)
from lerobot.common.robot_devices.robots.configs import So100RobotConfig
from lerobot.common.robot_devices.utils import busy_wait
from lerobot.common.utils.utils import get_safe_torch_device, has_method
from lerobot.common.robot_devices.robots.utils import Robot, make_robot_from_config
from lerobot.common.robot_devices.utils import safe_disconnect
from lerobot.common.utils.utils import has_method, init_logging, log_say
from lerobot.configs import parser
from lerobot.common.policies.act.configuration_act import (
    ACTConfig,
    NormalizationMode,
)
from lerobot.configs.types import PolicyFeature, FeatureType
import subprocess

import torch


########################################################################################
# Control modes
########################################################################################

NUM_COLS = 8
NUM_ROWS = 3

@safe_disconnect
def teleoperate(robot: Robot, cfg: TeleoperateControlConfig):
    control_loop(
        robot,
        control_time_s=cfg.teleop_time_s,
        fps=cfg.fps,
        teleoperate=True,
        display_data=cfg.display_data,
    )


@dataclass
class Config:
    robot: So100RobotConfig
    control: RecordControlConfig

@safe_disconnect
def record(
    robot: Robot,
    cfg: RecordControlConfig,
    row_col: tuple[int, int] = None,
) -> LeRobotDataset:
    # Create empty dataset or load existing saved episodes
    sanity_check_dataset_name(cfg.repo_id, cfg.policy)
    dataset = LeRobotDataset.create(
        cfg.repo_id,
        cfg.fps,
        root=cfg.root,
        robot=robot,
        use_videos=cfg.video,
        image_writer_processes=cfg.num_image_writer_processes,
        image_writer_threads=cfg.num_image_writer_threads_per_camera * len(robot.cameras),
    )

    # Load pretrained policy
    print(f"cfg policy: {cfg.policy}")
    policy = None if cfg.policy is None else make_policy(cfg.policy, ds_meta=dataset.meta)

    listener,events = init_keyboard_listener()
    if not robot.is_connected:
        robot.connect()
    robot.send_action(torch.tensor([0, 135, 135, 4, -90, 3]))

    # Execute a few seconds without recording to:
    # 1. teleoperate the robot to move it in starting position if no policy provided,
    # 2. give times to the robot devices to connect and start synchronizing,
    # 3. place the cameras windows on screen
    enable_teleoperation = policy is None
    warmup_record(robot, events, enable_teleoperation, cfg.warmup_time_s, cfg.display_data, cfg.fps)

    if has_method(robot, "teleop_safety_stop"):
        robot.teleop_safety_stop()


    recorded_episodes = 0
    control_time_s = cfg.episode_time_s
    while True:
        if recorded_episodes >= cfg.num_episodes:
            break
        
        current_grid = torch.tensor(row_col, dtype=torch.float)

        log_say(f"Recording episode {dataset.num_episodes}", cfg.play_sounds)
        if not robot.is_connected:
            robot.connect()

        if events is None:
            events = {"exit_early": False}

        if control_time_s is None:
            control_time_s = float("inf")

        if dataset is not None and cfg.single_task is None:
            raise ValueError("You need to provide a task as argument in `single_task`.")

        if dataset is not None and cfg.fps is not None and dataset.fps != cfg.fps:
            raise ValueError(f"The dataset fps should be equal to requested fps ({dataset['fps']} != {cfg.fps}).")
        
        timestamp = 0
        start_episode_t = time.perf_counter()
        while timestamp < control_time_s:
            start_loop_t = time.perf_counter()
            
            observation = robot.capture_observation()
            observation["grid_position"] = current_grid

            if policy is not None:
                pred_action = predict_action(
                    observation, policy, get_safe_torch_device(policy.config.device), policy.config.use_amp
                )
                # Action can eventually be clipped using `max_relative_target`,
                # so action actually sent is saved in the dataset.
                action = robot.send_action(pred_action)
                action = {"action": action}

            if dataset is not None:
                frame = {**observation, **action, "task": cfg.single_task, "grid_position": current_grid}
                dataset.add_frame(frame)

            if (cfg.display_data and not is_headless()):
                for k, v in action.items():
                    for i, vv in enumerate(v):
                        rr.log(f"sent_{k}_{i}", rr.Scalar(vv.numpy()))

                image_keys = [key for key in observation if "image" in key]
                for key in image_keys:
                    rr.log(key, rr.Image(observation[key].numpy()), static=True)

            if cfg.fps is not None:
                dt_s = time.perf_counter() - start_loop_t
                busy_wait(1 / cfg.fps - dt_s)

            dt_s = time.perf_counter() - start_loop_t
            log_control_info(robot, dt_s, fps=cfg.fps)

            timestamp = time.perf_counter() - start_episode_t
            if events["exit_early"]:
                events["exit_early"] = False
                break

        if not events["stop_recording"] and (
            (recorded_episodes < cfg.num_episodes - 1) or events["rerecord_episode"]
        ):
            log_say("Reset the environment", cfg.play_sounds)
            reset_environment(robot, events, cfg.reset_time_s, cfg.fps)

        if events["rerecord_episode"]:
            log_say("Re-record episode", cfg.play_sounds)
            events["rerecord_episode"] = False
            events["exit_early"] = False
            dataset.clear_episode_buffer()
            continue

        dataset.save_episode()
        recorded_episodes += 1

        if events["stop_recording"]:
            break

    log_say("Stop recording", cfg.play_sounds, blocking=True)
    stop_recording(robot, listener, cfg.display_data)

    if cfg.push_to_hub:
        dataset.push_to_hub(tags=cfg.tags, private=cfg.private)

    log_say("Exiting", cfg.play_sounds)
    return dataset

@dataclass
class Config_dummy():
    robot: So100RobotConfig
    control: RecordControlConfig

# @parser.wrap()
def control_robot(
    # cfg: ControlPipelineConfig,
    row_col: tuple[int, int],
):

    cfg = Config_dummy(
        robot=So100RobotConfig(
            leader_arms={
                'main': FeetechMotorsBusConfig(
                    port='/dev/ttyACM1',
                    motors={
                        'shoulder_pan': [1, 'sts3215'],
                        'shoulder_lift': [2, 'sts3215'],
                        'elbow_flex': [3, 'sts3215'],
                        'wrist_flex': [4, 'sts3215'],
                        'wrist_roll': [5, 'sts3215'],
                        'gripper': [6, 'sts3215']
                    },
                    mock=False
                )
            },
            follower_arms={
                'main': FeetechMotorsBusConfig(
                    port='/dev/ttyACM0',
                    motors={
                        'shoulder_pan': [1, 'sts3215'],
                        'shoulder_lift': [2, 'sts3215'],
                        'elbow_flex': [3, 'sts3215'],
                        'wrist_flex': [4, 'sts3215'],
                        'wrist_roll': [5, 'sts3215'],
                        'gripper': [6, 'sts3215']
                    },
                    mock=False
                )
            },
            cameras={
                'mounted': OpenCVCameraConfig(
                    camera_index=4,
                    fps=30,
                    width=640,
                    height=480,
                    color_mode='rgb',
                    channels=3,
                    rotation=None,
                    mock=False
                )
            },
            max_relative_target=None,
            gripper_open_degree=None,
            mock=False,
            calibration_dir='/home/achapin/hackathon/lerobot_jds/backend/lerobot/.cache/calibration/so100'
        ), 
        control=RecordControlConfig(
            repo_id='lirislab/eval_act_guess_who_33',
            single_task='',
            collect_grid=True,
            root=None,
            policy=ACTConfig(
                n_obs_steps=1,
                normalization_mapping={
                    'VISUAL': NormalizationMode.MEAN_STD,
                    'STATE': NormalizationMode.MEAN_STD,
                    'ACTION': NormalizationMode.MEAN_STD
                },
                input_features={
                    'observation.state': PolicyFeature(type=FeatureType.STATE, shape=(6,)),
                    'observation.images.mounted': PolicyFeature(type=FeatureType.VISUAL, shape=(3, 480, 640))
                },
                output_features={
                    'action': PolicyFeature(type=FeatureType.ACTION, shape=(6,))
                },
                device='cuda',
                use_amp=False,
                chunk_size=100,
                n_action_steps=100,
                use_grid=True,
                vision_backbone='resnet18',
                pretrained_backbone_weights='ResNet18_Weights.IMAGENET1K_V1',
                replace_final_stride_with_dilation=0,
                pre_norm=False,
                dim_model=512,
                n_heads=8,
                dim_feedforward=3200,
                feedforward_activation='relu',
                n_encoder_layers=4,
                n_decoder_layers=1,
                use_vae=True,
                latent_dim=32,
                n_vae_encoder_layers=4,
                temporal_ensemble_coeff=None,
                dropout=0.1,
                kl_weight=10.0,
                optimizer_lr=1e-05,
                optimizer_weight_decay=0.0001,
                optimizer_lr_backbone=1e-05,
                pretrained_path='/home/achapin/hackathon/lerobot_jds/backend/src/features/guess_who/checkpoints/100000_full_light/pretrained_model'
            ),
            fps=30,
            warmup_time_s=5,
            episode_time_s=30,
            reset_time_s=10,
            num_episodes=1,
            video=True,
            push_to_hub=False,
            private=False,
            tags=[''],
            num_image_writer_processes=0,
            num_image_writer_threads_per_camera=4,
            display_data=False,
            play_sounds=True,
            resume=False
        )
    )
    cfg.control.policy.pretrained_path = '/home/achapin/hackathon/lerobot_jds/backend/src/features/guess_who/checkpoints/050000_full/pretrained_model'
    init_logging()
    logging.info(pformat(asdict(cfg)))

    robot = make_robot_from_config(cfg.robot)
    record(robot, cfg.control, row_col=row_col)

    if robot.is_connected:
        robot.disconnect()


def robot_move_grid(row: int, col: int):
    """
    Moves the robot to a specific grid position.
    
    Args:
        row (int): The row index of the grid.
        col (int): The column index of the grid.
    """
    control_robot(row_col=[row, col])

if __name__ == "__main__":
   for (i, j) in [(1,6)]:
            print(f"Moving to grid position: ({i}, {j})")
            control_robot(row_col=[i, j])
