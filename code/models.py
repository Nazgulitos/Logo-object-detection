from typing import List, Tuple

import numpy as np
from PIL import Image

try:
    from ultralytics import YOLO
except ImportError:  # pragma: no cover
    YOLO = None  # type: ignore


class YoloDetector:
    """YOLOv8 detector wrapper for T-Bank logo detection.

    Exposes a minimal interface returning absolute pixel bboxes.
    """

    def __init__(self, weights_path: str, device: str | None = None, conf_threshold: float = 0.25):
        if YOLO is None:
            raise RuntimeError("Ultralytics is not installed. Please ensure 'ultralytics' is in requirements.")

        self.model = YOLO(weights_path)
        # Device can be 'cpu', 'cuda', 'mps', or None for auto
        self.device = device
        self.conf_threshold = conf_threshold

    def predict(self, image: Image.Image) -> List[Tuple[int, int, int, int]]:
        """Run inference and return list of (x_min, y_min, x_max, y_max) in pixels.

        Only class 0 ('logo') is returned as per data.yaml.
        """
        if image.mode not in ("RGB", "RGBA"):
            image = image.convert("RGB")

        # Run model
        results = self.model.predict(
            image, conf=self.conf_threshold, device=self.device, verbose=False
        )

        if not results:
            return []

        result = results[0]

        boxes_out: List[Tuple[int, int, int, int]] = []
        if result.boxes is None:
            return boxes_out

        # Ultralytics returns xyxy in float tensor
        xyxy = result.boxes.xyxy.cpu().numpy() if hasattr(result.boxes.xyxy, "cpu") else np.array(result.boxes.xyxy)
        clss = (
            result.boxes.cls.cpu().numpy().astype(int)
            if hasattr(result.boxes.cls, "cpu")
            else np.asarray(result.boxes.cls, dtype=int)
        )
        confs = (
            result.boxes.conf.cpu().numpy()
            if hasattr(result.boxes.conf, "cpu")
            else np.asarray(result.boxes.conf)
        )

        width, height = image.size

        for (x1, y1, x2, y2), cls_id, _ in zip(xyxy, clss, confs):
            # Keep only class 0 (logo). Ignore any other classes if present.
            if cls_id != 0:
                continue
            # Clip to image bounds and cast to int
            x_min = int(max(0, min(x1, x2)))
            y_min = int(max(0, min(y1, y2)))
            x_max = int(min(width - 1, max(x1, x2)))
            y_max = int(min(height - 1, max(y1, y2)))
            if x_max > x_min and y_max > y_min:
                boxes_out.append((x_min, y_min, x_max, y_max))

        return boxes_out


