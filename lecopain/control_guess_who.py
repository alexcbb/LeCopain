import logging
import os
import time
from dataclasses import asdict
from pprint import pformat

import rerun as rr

# from safetensors.torch import load_file, save_file
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from lerobot.common.policies.factory import make_policy
from lerobot.common.robot_devices.control_configs import (
    ControlConfig,
    ControlPipelineConfig,
    RecordControlConfig,
    RemoteRobotConfig,
    TeleoperateControlConfig,
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
from lerobot.common.robot_devices.utils import busy_wait
from lerobot.common.utils.utils import get_safe_torch_device, has_method
from lerobot.common.robot_devices.robots.utils import Robot, make_robot_from_config
from lerobot.common.robot_devices.utils import safe_disconnect
from lerobot.common.utils.utils import has_method, init_logging, log_say
from lerobot.configs import parser
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
    policy = None if cfg.policy is None else make_policy(cfg.policy, ds_meta=dataset.meta)

    if not robot.is_connected:
        robot.connect()
    robot.send_action(torch.tensor([0, 135, 135, 4, -90, 3]))

    listener, events = init_keyboard_listener()

    # Execute a few seconds without recording to:
    # 1. teleoperate the robot to move it in starting position if no policy provided,
    # 2. give times to the robot devices to connect and start synchronizing,
    # 3. place the cameras windows on screen
    enable_teleoperation = policy is None
    log_say("Warmup record", cfg.play_sounds)
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

def _init_rerun(control_config: ControlConfig, session_name: str = "lerobot_control_loop") -> None:
    """Initializes the Rerun SDK for visualizing the control loop.

    Args:
        control_config: Configuration determining data display and robot type.
        session_name: Rerun session name. Defaults to "lerobot_control_loop".

    Raises:
        ValueError: If viewer IP is missing for non-remote configurations with display enabled.
    """
    if (control_config.display_data and not is_headless()) or (
        control_config.display_data and isinstance(control_config, RemoteRobotConfig)
    ):
        # Configure Rerun flush batch size default to 8KB if not set
        batch_size = os.getenv("RERUN_FLUSH_NUM_BYTES", "8000")
        os.environ["RERUN_FLUSH_NUM_BYTES"] = batch_size

        # Initialize Rerun based on configuration
        rr.init(session_name)
        if isinstance(control_config, RemoteRobotConfig):
            viewer_ip = control_config.viewer_ip
            viewer_port = control_config.viewer_port
            if not viewer_ip or not viewer_port:
                raise ValueError(
                    "Viewer IP & Port are required for remote config. Set via config file/CLI or disable control_config.display_data."
                )
            logging.info(f"Connecting to viewer at {viewer_ip}:{viewer_port}")
            rr.connect_tcp(f"{viewer_ip}:{viewer_port}")
        else:
            # Get memory limit for rerun viewer parameters
            memory_limit = os.getenv("LEROBOT_RERUN_MEMORY_LIMIT", "10%")
            rr.spawn(memory_limit=memory_limit)

@parser.wrap()
def control_robot(
    cfg: ControlPipelineConfig,
    row_col: tuple[int, int] = None,
):

    init_logging()
    logging.info(pformat(asdict(cfg)))

    robot = make_robot_from_config(cfg.robot)
    _init_rerun(control_config=cfg.control, session_name="lerobot_control_loop_record")
    record(robot, cfg.control, row_col=row_col)

    if robot.is_connected:
        robot.disconnect()

# The following function is a placeholder for the robot's movement logic.
# It is the function called by the backend (Mistral) to move the robot to a specific grid position.
def robot_move_grid(row: int, col: int):
    """
    Moves the robot to a specific grid position.
    
    Args:
        row (int): The row index of the grid.
        col (int): The column index of the grid.
    """
    control_robot(row_col=[row, col])

if __name__ == "__main__":
    control_robot(row_col=[0, 2])