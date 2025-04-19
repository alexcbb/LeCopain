from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
import torch

NUM_COLS = 8
NUM_ROWS = 3
if __name__ == "__main__":
    dataset = LeRobotDataset(
        repo_id="<user-id>/<repo-id>",
    )
    dataset.push_to_hub()