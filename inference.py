import os
import argparse
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
from PIL import Image

import torch

from model import build_model


NUM_CLASSES = 1
KNOWN_SOURCES = ["dataset", "dataset2", "dataset3", "pb_extra_dataset"]

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

CLASS_COLORS = {
    0: (0, 0, 0),         # background = black
    1: (255, 0, 0),       # building   = red
}

VALID_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def infer_source_from_path(path):
    norm = str(path).replace("\\", "/")
    parts = norm.split("/")
    for src in KNOWN_SOURCES:
        if src in parts or f"/{src}/" in norm:
            return src
    return "unknown"


def detect_image_column(df):
    cols = list(df.columns)
    lower_map = {c.lower(): c for c in cols}
    image_candidates = ["image", "image_path", "img", "img_path", "images", "image_file", "input"]

    for c in image_candidates:
        if c in lower_map:
            return lower_map[c]

    if len(cols) >= 1:
        return cols[0]

    raise ValueError(f"Could not detect image column from CSV columns: {cols}")


def load_checkpoint(model, checkpoint_path, device):
    ckpt = torch.load(checkpoint_path, map_location=device)

    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        state_dict = ckpt["model_state_dict"]
    elif isinstance(ckpt, dict) and "state_dict" in ckpt:
        state_dict = ckpt["state_dict"]
    else:
        state_dict = ckpt

    missing, unexpected = model.load_state_dict(state_dict, strict=False)

    print("\n[Checkpoint Load Info]")
    print(f"Loaded from: {checkpoint_path}")
    print(f"Missing keys: {len(missing)}")
    print(f"Unexpected keys: {len(unexpected)}")

    if len(missing) > 0:
        print("First missing keys:", missing[:10])
    if len(unexpected) > 0:
        print("First unexpected keys:", unexpected[:10])


def preprocess_image(image_pil):
    image_np = np.array(image_pil).astype(np.float32) / 255.0
    image_np = (image_np - IMAGENET_MEAN) / IMAGENET_STD
    image_np = np.transpose(image_np, (2, 0, 1))
    return torch.tensor(image_np, dtype=torch.float32).unsqueeze(0)


def colorize_mask(mask):
    h, w = mask.shape
    color_mask = np.zeros((h, w, 3), dtype=np.uint8)
    for class_id, color in CLASS_COLORS.items():
        color_mask[mask == class_id] = color
    return color_mask


def create_overlay(original_rgb, color_mask, alpha=0.45):
    original_rgb = original_rgb.astype(np.float32)
    color_mask = color_mask.astype(np.float32)
    overlay = (1 - alpha) * original_rgb + alpha * color_mask
    return np.clip(overlay, 0, 255).astype(np.uint8)


def gather_images_from_folder(folder_path):
    folder = Path(folder_path)
    if not folder.exists():
        raise FileNotFoundError(f"Folder not found: {folder_path}")

    image_paths = []
    for ext in VALID_EXTENSIONS:
        image_paths.extend(folder.glob(f"*{ext}"))
        image_paths.extend(folder.glob(f"*{ext.upper()}"))

    image_paths = sorted(set(str(p) for p in image_paths))
    if not image_paths:
        raise FileNotFoundError(f"No supported image files found in: {folder_path}")

    return image_paths


def gather_images_from_csv(csv_path, max_per_source=None, seed=42):
    df = pd.read_csv(csv_path)
    image_col = detect_image_column(df)

    records = []
    for _, row in df.iterrows():
        image_path = str(row[image_col])
        source = infer_source_from_path(image_path)
        records.append({"image_path": image_path, "source": source})

    if max_per_source is None:
        return records

    rng = np.random.default_rng(seed)
    grouped = defaultdict(list)
    for r in records:
        grouped[r["source"]].append(r)

    sampled = []
    for source, items in grouped.items():
        n = min(max_per_source, len(items))
        idxs = rng.choice(len(items), size=n, replace=False)
        sampled.extend([items[i] for i in idxs])

    return sampled


@torch.no_grad()
def predict_mask(model, image_path, device):
    image_pil = Image.open(image_path).convert("RGB")
    original_np = np.array(image_pil)

    input_tensor = preprocess_image(image_pil).to(device)
    logits = model(input_tensor)
    pred_mask = torch.argmax(logits, dim=1).squeeze(0).cpu().numpy().astype(np.uint8)

    color_mask = colorize_mask(pred_mask)
    overlay = create_overlay(original_np, color_mask, alpha=0.45)

    return original_np, pred_mask, color_mask, overlay


def save_prediction_outputs(image_path, source, original_np, pred_mask, color_mask, overlay, output_dir):
    image_name = Path(image_path).stem

    source_dir = os.path.join(output_dir, source)
    orig_dir = os.path.join(source_dir, "original")
    raw_dir = os.path.join(source_dir, "raw_mask")
    color_dir = os.path.join(source_dir, "color_mask")
    overlay_dir = os.path.join(source_dir, "overlay")

    ensure_dir(orig_dir)
    ensure_dir(raw_dir)
    ensure_dir(color_dir)
    ensure_dir(overlay_dir)

    orig_path = os.path.join(orig_dir, f"{image_name}.png")
    raw_path = os.path.join(raw_dir, f"{image_name}.png")
    color_path = os.path.join(color_dir, f"{image_name}.png")
    overlay_path = os.path.join(overlay_dir, f"{image_name}.png")

    Image.fromarray(original_np).save(orig_path)
    Image.fromarray(pred_mask).save(raw_path)
    Image.fromarray(color_mask).save(color_path)
    Image.fromarray(overlay).save(overlay_path)

    return orig_path, raw_path, color_path, overlay_path


def main():
    parser = argparse.ArgumentParser(description="Visual inference for geospatial segmentation")
    parser.add_argument(
        "--input_mode",
        type=str,
        choices=["folder", "csv"],
        required=True,
        help="Use 'folder' for a dataset folder or 'csv' for sampling from CSV."
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Folder path or CSV path depending on input_mode."
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="best_model_unetpp.pth",
        help="Path to checkpoint."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="prediction_outputs",
        help="Output directory."
    )
    parser.add_argument(
        "--encoder_name",
        type=str,
        default="resnet34",
        help="Encoder used during training."
    )
    parser.add_argument(
        "--max_per_source",
        type=int,
        default=5,
        help="Only used in csv mode. Samples this many images per source."
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for CSV sampling."
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")

    model = build_model(
        num_classes=NUM_CLASSES,
        encoder_name=args.encoder_name,
        encoder_weights=None
    ).to(device)

    load_checkpoint(model, args.checkpoint, device)
    model.eval()

    ensure_dir(args.output_dir)

    if args.input_mode == "folder":
        image_paths = gather_images_from_folder(args.input)
        records = [{"image_path": p, "source": infer_source_from_path(p)} for p in image_paths]
    else:
        records = gather_images_from_csv(
            csv_path=args.input,
            max_per_source=args.max_per_source,
            seed=args.seed
        )

    print(f"Total images selected: {len(records)}")

    for idx, rec in enumerate(records, start=1):
        image_path = rec["image_path"]
        source = rec["source"]

        try:
            original_np, pred_mask, color_mask, overlay = predict_mask(model, image_path, device)
            orig_path, raw_path, color_path, overlay_path = save_prediction_outputs(
                image_path=image_path,
                source=source,
                original_np=original_np,
                pred_mask=pred_mask,
                color_mask=color_mask,
                overlay=overlay,
                output_dir=args.output_dir
            )

            print(f"[{idx}/{len(records)}] {source} -> {image_path}")
            print(f"  original : {orig_path}")
            print(f"  raw_mask : {raw_path}")
            print(f"  color    : {color_path}")
            print(f"  overlay  : {overlay_path}")

        except Exception as e:
            print(f"[{idx}/{len(records)}] FAILED -> {image_path}")
            print(f"  Error: {e}")

    print("\nInference complete.")


if __name__ == "__main__":
    main()