#!/usr/bin/env python3
"""
Test action prediction using the SmolVLA policy.
Loads a dataset sample from 'lerobot/svla_so100_pickplace', maps the camera views,
and generates real 6-DoF robot motor actions on CPU or GPU.
"""

import sys
from pathlib import Path
import torch

# Ensure local smolvla package can be loaded directly from the repository
repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from lerobot.datasets.lerobot_dataset import LeRobotDataset

try:
    from smolvla import SmolVLAPolicy, make_smolvla_pre_post_processors
except ImportError:
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
    from lerobot.policies.smolvla.processor_smolvla import make_smolvla_pre_post_processors


def predict(
    model_id: str = "lerobot/smolvla_base",
    dataset_id: str = "lerobot/svla_so100_pickplace",
    sample_idx: int = 0,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
):
    print(f"Using device: {device.upper()}")

    print(f"1. Loading dataset sample from '{dataset_id}'...")
    dataset = LeRobotDataset(dataset_id)
    sample = dataset[sample_idx]

    print(f"2. Loading SmolVLA model from '{model_id}'...")
    policy = SmolVLAPolicy.from_pretrained(model_id)
    policy.to(device)
    policy.eval()

    # Map dataset camera names ('top', 'wrist') to policy camera names ('camera1', 'camera2')
    rename_map = {
        "observation.images.top": "observation.images.camera1",
        "observation.images.wrist": "observation.images.camera2",
    }

    print("3. Setting up pre/post processing pipelines...")
    preprocessor, postprocessor = make_smolvla_pre_post_processors(
        config=policy.config,
        dataset_stats=dataset.meta.stats,
    )

    for step in preprocessor.steps:
        if hasattr(step, "rename_map"):
            step.rename_map = rename_map

    print("4. Preprocessing input sample...")
    model_inputs = preprocessor(sample)

    print("5. Predicting action chunk with SmolVLA...")
    with torch.no_grad():
        action_chunk = policy.select_action(model_inputs)
        action_chunk = postprocessor(action_chunk)

    print("\n" + "=" * 45)
    print("        PREDICTION SUCCESSFUL!")
    print("=" * 45)
    print(f"Task Instruction: '{sample['task']}'")
    print(f"Action Chunk Shape: {action_chunk.shape} (timesteps x motor DOFs)")
    print("\nPredicted 6-DoF Motor Angles (First 3 Timesteps in degrees):")
    print(action_chunk[:3])


if __name__ == "__main__":
    predict()
