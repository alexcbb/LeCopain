from huggingface_hub import HfApi
import logging
import pandas as pd
from tqdm import tqdm
import json
import os
import argparse

from lerobot.common.datasets.lerobot_dataset import LeRobotDatasetMetadata

def fetch_lerobot_datasets(api: HfApi):
    datasets = api.list_datasets(tags="LeRobot")
    repo_ids = [dataset.id for dataset in datasets]
    return repo_ids

def analyze_dataset_metadata(repo_id: str, api: HfApi):
    try:
        metadata = LeRobotDatasetMetadata(repo_id=repo_id, revision="v2.0")
    except Exception as e:
        try:
            metadata = LeRobotDatasetMetadata(repo_id=repo_id, revision="v2.1")
        except Exception as e:
            print(f"Error loading metadata for {repo_id}: {str(e)}")
            return None

    # Check version
    version_str = str(metadata._version).strip()
    if version_str not in ["2.0", "2.1"]:
        print(f"Skipping {repo_id}: version <{version_str}>")
        return None

    try:
        info = {
            "repo_id": repo_id,
            "username": repo_id.split('/')[0],
            "robot_type": metadata.robot_type,
            "total_episodes": metadata.total_episodes,
            "total_frames": metadata.total_frames,
            "fps": metadata.fps,
            "camera_keys": ','.join(metadata.camera_keys),  # Convert list to string
            "num_cameras": len(metadata.camera_keys),
            "video_keys": ','.join(metadata.video_keys),
            "has_video": len(metadata.video_keys) > 0,
            "total_tasks": metadata.total_tasks,
            "tasks": json.dumps(metadata.tasks),  # Convert dict to JSON string
            "is_sim": "sim_" in repo_id.lower(),
            "is_eval": "eval_" in repo_id.lower(),
            "features": ','.join(metadata.features.keys()),
            "chunks_size": metadata.chunks_size,
            "total_chunks": metadata.total_chunks,
            "version": metadata._version,
            "creation_date": api.repo_info(repo_id=repo_id, repo_type="dataset").last_modified
        }
        return info
    except Exception as e:
        print(f"Error extracting metadata for {repo_id}: {str(e)}")
        return None

if __name__ == "__main__":
    logging.disable(logging.CRITICAL)
    parser=argparse.ArgumentParser()
    parser.add_argument('--file', type=str, default='', help='File to load')
    args=parser.parse_args()

    api = HfApi()
    # Get the list of dataset repo_ids
    lerobot_datasets = fetch_lerobot_datasets(api)
    print(f"Total LeRobot datasets found: {len(lerobot_datasets)}")

    file = args.file
    if file == '' or not file.endswith('.csv') or not os.path.exists(file):
        print(f"File {file} does not exist")
    if file:
        df = pd.read_csv(file)
        existing_repo_ids = df['repo_id'].values
    else:
        df = pd.DataFrame()
        existing_repo_ids = []
    print(f"Loaded {len(df)} datasets from {file}")
    print(f"There is {len(lerobot_datasets) - len(df)} datasets to analyze")

    lerobot_datasets = [repo_id for repo_id in lerobot_datasets if repo_id not in existing_repo_ids]
    # Collect all dataset info
    dataset_infos = []
    for repo_id in tqdm(lerobot_datasets):
        info = analyze_dataset_metadata(repo_id, api)
        if info is not None:
            dataset_infos.append(info)

    # Convert to DataFrame and save to CSV
    new_df = pd.DataFrame(dataset_infos)

    df = pd.concat([df, new_df], ignore_index=True)

    # Save to CSV
    csv_filename = "lerobot_datasets_last.csv"
    df.to_csv(csv_filename, index=False)

    # Print summary
    print("\nDataset Summary:")
    print("-" * 40)
    print(f"Total datasets analyzed: {len(df)}")
    print(f"\nRobot types distribution:")
    print(df['robot_type'].value_counts())
    print(f"\nDataset saved to: {csv_filename}")

    # Additional useful stats
    print(f"\nDatasets with video: {df['has_video'].sum()}")
    print(f"Simulation datasets: {df['is_sim'].sum()}")
    print(f"Evaluation datasets: {df['is_eval'].sum()}")
    print(f"\nAverage episodes per dataset: {df['total_episodes'].mean():.1f}")
    print(f"Average frames per dataset: {df['total_frames'].mean():.1f}")