#!/usr/bin/env python3
"""
Visualize a sample from the SO-100 Pick & Place dataset (lerobot/svla_so100_pickplace).
Extracts the top-down camera view and wrist camera view, stitches them side-by-side,
and annotates with task instructions and robot joint angles.
"""

from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import torchvision.transforms.functional as TF
from lerobot.datasets.lerobot_dataset import LeRobotDataset


def visualize_sample(repo_id: str = "lerobot/svla_so100_pickplace", sample_idx: int = 0, output_path: str = "sample_visualization.png"):
    print(f"Loading dataset '{repo_id}'...")
    dataset = LeRobotDataset(repo_id)

    print(f"Extracting sample index {sample_idx} (Total samples: {len(dataset)})...")
    sample = dataset[sample_idx]

    # Convert PyTorch tensors (C, H, W in [0, 1]) to PIL Images
    img_top = TF.to_pil_image(sample["observation.images.top"])
    img_wrist = TF.to_pil_image(sample["observation.images.wrist"])

    # Create composite canvas
    width_each, height_each = img_top.size
    header_height = 80
    combined_width = width_each * 2
    combined_height = height_each + header_height

    combined_img = Image.new("RGB", (combined_width, combined_height), color=(24, 24, 27))
    combined_img.paste(img_top, (0, header_height))
    combined_img.paste(img_wrist, (width_each, header_height))

    draw = ImageDraw.Draw(combined_img)
    font = ImageFont.load_default()

    task_text = f"Task: {sample.get('task', 'N/A')}"
    episode_idx = sample.get('episode_index', 0)
    frame_idx = sample.get('frame_index', 0)
    if hasattr(episode_idx, "item"):
        episode_idx = episode_idx.item()
    if hasattr(frame_idx, "item"):
        frame_idx = frame_idx.item()

    joint_angles = [round(x, 2) for x in sample['observation.state'].tolist()]
    meta_text = f"Episode: {episode_idx} | Frame: {frame_idx} | Motor State: {joint_angles}"

    # Draw header text & subtitles
    draw.text((20, 15), task_text, fill=(255, 255, 255), font=font)
    draw.text((20, 45), meta_text, fill=(180, 180, 180), font=font)
    draw.text((20, header_height + 10), "[Camera: TOP]", fill=(0, 255, 128), font=font)
    draw.text((width_each + 20, header_height + 10), "[Camera: WRIST]", fill=(0, 255, 128), font=font)

    # Save image
    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    combined_img.save(str(out_file))
    print(f"Visualization saved successfully to: {out_file.resolve()}")


if __name__ == "__main__":
    visualize_sample()
