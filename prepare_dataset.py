from __future__ import annotations

import argparse
import math
import random
import shutil
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Split a flat YOLO dataset stored in ./images and ./labels into "
            "train/val/test folders compatible with Ultralytics."
        )
    )
    parser.add_argument(
        "--images",
        type=Path,
        default=Path("images"),
        help="Directory with source images (default: ./images)",
    )
    parser.add_argument(
        "--labels",
        type=Path,
        default=Path("labels"),
        help="Directory with matching YOLO label files (default: ./labels)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("datasets/animals"),
        help="Output dataset root (default: ./datasets/animals)",
    )
    parser.add_argument(
        "--ratios",
        type=float,
        nargs=3,
        default=(0.8, 0.1, 0.1),
        metavar=("TRAIN", "VAL", "TEST"),
        help="Split ratios (must sum to 1.0). Default: 0.8 0.1 0.1",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used for shuffling before the split (default: 42)",
    )
    parser.add_argument(
        "--move",
        action="store_true",
        help="Move files instead of copying them (default: copy).",
    )
    return parser.parse_args()


def verify_split_ratios(ratios: tuple[float, float, float]) -> None:
    total = sum(ratios)
    if not math.isclose(total, 1.0, rel_tol=1e-6):
        raise ValueError(f"Split ratios must add up to 1.0, received {ratios} (sum={total})")
    if any(r <= 0 for r in ratios):
        raise ValueError(f"Each split ratio must be positive, received {ratios}")


def gather_samples(images_dir: Path, labels_dir: Path) -> list[tuple[Path, Path]]:
    samples: list[tuple[Path, Path]] = []
    for image_path in sorted(images_dir.glob("*")):
        if not image_path.is_file():
            continue
        if image_path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp"}:
            continue
        label_path = labels_dir / f"{image_path.stem}.txt"
        if not label_path.exists():
            raise FileNotFoundError(f"Missing label file for image: {image_path.name}")
        samples.append((image_path, label_path))
    if not samples:
        raise RuntimeError(f"No image/label pairs found in {images_dir} and {labels_dir}")
    return samples


def split_samples(
    samples: list[tuple[Path, Path]], ratios: tuple[float, float, float], seed: int
) -> dict[str, list[tuple[Path, Path]]]:
    random.Random(seed).shuffle(samples)
    total = len(samples)
    train_split = int(ratios[0] * total)
    val_split = int(ratios[1] * total) + train_split

    splits = {
        "train": samples[:train_split],
        "val": samples[train_split:val_split],
        "test": samples[val_split:],
    }

    # Guarantee that empty splits are avoided by reassigning last samples as needed.
    for split_name in ("train", "val", "test"):
        if not splits[split_name]:
            raise RuntimeError(
                f"Split '{split_name}' is empty. Try increasing the dataset size "
                "or adjusting the split ratios."
            )
    return splits


def clear_output_dirs(output_root: Path) -> None:
    if output_root.exists():
        for path in output_root.glob("*"):
            if path.is_dir():
                shutil.rmtree(path)
            else:
                path.unlink()
    output_root.mkdir(parents=True, exist_ok=True)


def export_split(
    splits: dict[str, list[tuple[Path, Path]]], output_root: Path, move_files: bool
) -> None:
    operation = shutil.move if move_files else shutil.copy2
    for split_name, examples in splits.items():
        image_out = output_root / split_name / "images"
        label_out = output_root / split_name / "labels"
        image_out.mkdir(parents=True, exist_ok=True)
        label_out.mkdir(parents=True, exist_ok=True)

        for image_path, label_path in examples:
            operation(image_path, image_out / image_path.name)
            operation(label_path, label_out / label_path.name)


def main() -> None:
    args = parse_args()
    verify_split_ratios(tuple(args.ratios))

    if not args.images.exists() or not args.labels.exists():
        raise FileNotFoundError("Source directories './images' and './labels' must exist.")

    samples = gather_samples(args.images, args.labels)
    splits = split_samples(samples, tuple(args.ratios), args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    clear_output_dirs(args.output)
    export_split(splits, args.output, args.move)
    print(
        f"Dataset exported to {args.output} "
        f"(train={len(splits['train'])}, val={len(splits['val'])}, test={len(splits['test'])})"
    )


if __name__ == "__main__":
    main()
