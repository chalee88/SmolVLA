#!/usr/bin/env python3
"""
Inspect dataset structure and sample keys from the SO-100 Pick & Place dataset.
Displays metadata, frame count, episode count, feature signatures, and tensor shapes.
"""

import argparse
from lerobot.datasets.lerobot_dataset import LeRobotDataset


def inspect_dataset(repo_id: str = "lerobot/svla_so100_pickplace", sample_idx: int = 0):
    print(f"Loading dataset '{repo_id}'...")
    dataset = LeRobotDataset(repo_id)

    print("\n" + "=" * 50)
    print("           DATASET METADATA")
    print("=" * 50)
    print(f"Dataset Repository : {repo_id}")
    print(f"Total Frames       : {len(dataset):,}")
    print(f"Total Episodes     : {dataset.num_episodes:,}")
    print(f"Camera FPS         : {getattr(dataset, 'fps', 'N/A')}")

    print("\n" + "=" * 50)
    print(f"       SAMPLE INSPECTION (Index: {sample_idx})")
    print("=" * 50)
    sample = dataset[sample_idx]

    print(f"Task Instruction: '{sample.get('task', 'N/A')}'\n")
    print(f"{'Feature Key':<35} {'Details / Shape / Value'}")
    print("-" * 65)

    for key, val in sorted(sample.items()):
        if hasattr(val, "shape") and hasattr(val, "dtype"):
            print(f"{key:<35} shape = {str(list(val.shape)):<18} dtype = {val.dtype}")
        elif isinstance(val, (int, float, str, bool)):
            print(f"{key:<35} value = {val}")
        else:
            print(f"{key:<35} type  = {type(val).__name__}")

    print("\n" + "=" * 50)
    if "observation.state" in sample:
        motor_angles = [round(float(x), 3) for x in sample["observation.state"]]
        print(f"Current Motor Angles (6-DoF): {motor_angles}")
    print("=" * 50)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inspect LeRobot dataset structure for SmolVLA.")
    parser.add_argument("--repo_id", type=str, default="lerobot/svla_so100_pickplace", help="Hugging Face dataset repository ID")
    parser.add_argument("--sample_idx", type=int, default=0, help="Index of the frame sample to inspect")
    args = parser.parse_args()

    inspect_dataset(repo_id=args.repo_id, sample_idx=args.sample_idx)
