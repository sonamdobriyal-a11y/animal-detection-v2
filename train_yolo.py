from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a YOLO model on the animals dataset.")
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("animals.yaml"),
        help="Path to the dataset YAML file (default: animals.yaml).",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="yolov8n.pt",
        help="Ultralytics model checkpoint to fine-tune (default: yolov8n.pt).",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Number of training epochs (default: 100).",
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=16,
        help="Batch size used for training (default: 16).",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Image size used for resized training inputs (default: 640).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="",
        help="Device identifier, e.g. '0' for first GPU or 'cpu' (default: auto).",
    )
    parser.add_argument(
        "--project",
        type=Path,
        default=Path("runs/animals"),
        help="Directory where Ultralytics stores training artifacts (default: runs/animals).",
    )
    parser.add_argument(
        "--name",
        type=str,
        default="yolov8n-animals",
        help="Name of the training run (default: yolov8n-animals).",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume training if a previous run with the same project/name exists.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model = YOLO(args.model)

    results = model.train(
        data=str(args.data),
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        device=args.device,
        project=str(args.project),
        name=args.name,
        exist_ok=args.resume,
        resume=args.resume,
        pretrained=True,
    )

    print(results)


if __name__ == "__main__":
    main()
