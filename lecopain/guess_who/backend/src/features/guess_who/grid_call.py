import subprocess

import argparse
def parse_args():
    parser = argparse.ArgumentParser(description="Grid call arguments")
    parser.add_argument("--row", type=int, default=0, help="Row number")
    parser.add_argument("--col", type=int, default=4, help="Column number")
    return parser.parse_args()
def grid_call(row, col):
    result = subprocess.run(
        f"""python control_atomic.py --robot.type=so100   --control.type=record   --control.fps=30
        --control.single_task=""   --control.tags='[""]'   --control.warmup_time_s=5
        --control.episode_time_s=30   --control.reset_time_s=10   --control.num_episodes=1
        --control.push_to_hub=false   --control.policy.path=/home/achapin/hackathon/lerobot_jds/backend/src/features/guess_who/checkpoints/050000_full/pretrained_model/
        --control.repo_id=lirislab/eval_act_guess_who_34 --control.row={row} --control.col={col}""",
        shell=True, text=True, check=True, capture_output=True
    )
    print(f"{result.stdout}")
    print(f"{result.stderr}")
if __name__ == "__main__":
    args = parse_args()
    grid_call(args.row, args.col)
