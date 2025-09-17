from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

from PIL import Image

# Support running both as module (python -m code.validate) and as script (python code/validate.py)
try:
    from .models import YoloDetector  # type: ignore
except ImportError:  # pragma: no cover
    import sys
    import pathlib as _pathlib

    sys.path.append(str(_pathlib.Path(__file__).resolve().parent.parent))
    from code.models import YoloDetector  # type: ignore


@dataclass
class Metrics:
    precision: float
    recall: float
    f1: float


def compute_iou(box_a: Tuple[int, int, int, int], box_b: Tuple[int, int, int, int]) -> float:
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)

    union = area_a + area_b - inter_area
    if union == 0:
        return 0.0
    return inter_area / union


def match_detections(gt_boxes: List[Tuple[int, int, int, int]], pred_boxes: List[Tuple[int, int, int, int]], iou_thr: float) -> Tuple[int, int, int]:
    matched_gt = set()
    tp = 0
    for p in pred_boxes:
        # Find best matching GT above threshold
        best_iou = 0.0
        best_idx = -1
        for idx, g in enumerate(gt_boxes):
            if idx in matched_gt:
                continue
            iou = compute_iou(p, g)
            if iou > best_iou:
                best_iou = iou
                best_idx = idx
        if best_iou >= iou_thr and best_idx >= 0:
            tp += 1
            matched_gt.add(best_idx)
    fp = max(0, len(pred_boxes) - tp)
    fn = max(0, len(gt_boxes) - tp)
    return tp, fp, fn


def parse_label_file(label_path: Path, img_size: Tuple[int, int]) -> List[Tuple[int, int, int, int]]:
    width, height = img_size
    boxes: List[Tuple[int, int, int, int]] = []
    if not label_path.exists():
        return boxes
    for line in label_path.read_text().strip().splitlines():
        parts = line.strip().split()
        if len(parts) != 5:
            continue
        cls, cx, cy, w, h = parts
        # Keep only class 0
        if int(cls) != 0:
            continue
        cx = float(cx) * width
        cy = float(cy) * height
        bw = float(w) * width
        bh = float(h) * height
        x1 = int(cx - bw / 2)
        y1 = int(cy - bh / 2)
        x2 = int(cx + bw / 2)
        y2 = int(cy + bh / 2)
        boxes.append((x1, y1, x2, y2))
    return boxes


def evaluate(weights: str, images_dir: Path, labels_dir: Path, iou_thr: float = 0.5) -> Metrics:
    detector = YoloDetector(weights_path=weights)
    image_paths = sorted([p for p in images_dir.glob("*.jpg")] + [p for p in images_dir.glob("*.png")] + [p for p in images_dir.glob("*.bmp")] + [p for p in images_dir.glob("*.webp")])

    total_tp = total_fp = total_fn = 0
    for img_path in image_paths:
        with Image.open(img_path) as img:
            preds = detector.predict(img)
            gts = parse_label_file(labels_dir / (img_path.stem + ".txt"), img.size)
        tp, fp, fn = match_detections(gts, preds, iou_thr)
        total_tp += tp
        total_fp += fp
        total_fn += fn

    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    return Metrics(precision=precision, recall=recall, f1=f1)


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate YOLO detector on labeled dataset (YOLO format)")
    parser.add_argument("--weights", type=str, default="models/best-yolov8n.pt")
    parser.add_argument("--images", type=str, default="data/dataset/valid/images")
    parser.add_argument("--labels", type=str, default="data/dataset/valid/labels")
    parser.add_argument("--iou", type=float, default=0.5)
    args = parser.parse_args()

    metrics = evaluate(args.weights, Path(args.images), Path(args.labels), args.iou)
    print(json.dumps({"precision": metrics.precision, "recall": metrics.recall, "f1": metrics.f1}, indent=2))


if __name__ == "__main__":
    main()


