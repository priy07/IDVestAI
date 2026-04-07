"""
detector.py
-----------
IDVest AI — Standalone Hybrid Detector (mirrors app.detector.yolo_detector)

This file exists for backward compatibility. The canonical implementation
lives in app/detector/yolo_detector.py.

Usage:
    from detector import Detector
    det = Detector()
    results = det.detect(frame)
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional

import cv2
import numpy as np

try:
    from ultralytics import YOLO
    _YOLO_OK = True
except ImportError:
    _YOLO_OK = False
    print("[Detector] ERROR: ultralytics not installed. Run: pip install ultralytics")


class Detector:
    """
    Hybrid dual-model YOLO detector.

    Model 1 (yolov8n.pt)      → Person detection (COCO pretrained)
    Model 2 (idvest_best.pt)   → Domain-specific attire / ID detection

    Returns list of dicts: [{"label": str, "confidence": float, "box": (x1,y1,x2,y2)}]
    """

    CLASS_THRESHOLDS: Dict[str, float] = {
        "person":          0.50,
        "blazer":          0.30,
        "formals":         0.30,
        "faculty-casuals": 0.40,
        "id-card":         0.10,
        "faculty-id":      0.10,
    }

    ALLOWED_CLASSES = {
        "person",
        "id-card",
        "faculty-id",
        "formals",
        "blazer",
        "faculty-casuals"
    }

    def __init__(
        self,
        custom_model_path: str = "models/idvest_best.pt",
        confidence: float = 0.25,
    ) -> None:
        self.default_conf = confidence
        self._person_model = None
        self._custom_model = None

        if not _YOLO_OK:
            return

        print("[Detector] Loading Person Engine (yolov8n.pt)...")
        self._person_model = YOLO("yolov8n.pt")

        c_path = Path(custom_model_path)
        if c_path.exists():
            print(f"[Detector] Loading Custom Engine ({c_path.name})...")
            self._custom_model = YOLO(str(c_path))
        else:
            print(f"[Detector] WARNING: Custom weights missing at {c_path}.")

    def detect(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """Run hybrid detection pipeline with spatial filtering."""
        if self._person_model is None:
            return []

        # Phase 1: Person detection (COCO class 0)
        res_coco = self._person_model(source=frame, classes=[0], conf=0.45, imgsz=480, verbose=False)
        people = [
            d for d in self._parse_yolo(res_coco)
            if d["label"] == "person" and d["confidence"] >= self.CLASS_THRESHOLDS["person"]
        ]

        if not people:
            return []

        # Phase 2: Custom attire / ID detection
        final = list(people)

        if self._custom_model:
            res_items = self._custom_model(source=frame, conf=0.03, imgsz=640, verbose=False)
            raw_items = self._parse_yolo(res_items)

            for item in raw_items:
                item["label"] = item["label"].replace("_", "-")
                label = item["label"]

                if label not in self.ALLOWED_CLASSES:
                    continue

                is_id = label in ("id-card", "faculty-id")
                thresh = self.CLASS_THRESHOLDS.get(label, self.default_conf)
                if not is_id and item["confidence"] < thresh:
                    continue

                processed = self._spatial_filter(item, people)
                if processed:
                    final.append(processed)

        return final

    def _spatial_filter(
        self, item: Dict[str, Any], people: List[Dict[str, Any]]
    ) -> Optional[Dict[str, Any]]:
        """Accept item only if centroid is inside a person box. Boost ID confidence in chest."""
        ix = (item["box"][0] + item["box"][2]) / 2
        iy = (item["box"][1] + item["box"][3]) / 2
        margin = 15

        parent = None
        for p in people:
            px1, py1, px2, py2 = p["box"]
            if (px1 - margin <= ix <= px2 + margin) and (py1 - margin <= iy <= py2 + margin):
                parent = p
                break

        if parent is None:
            return None

        label = item["label"]
        conf = item["confidence"]

        if label in ("id-card", "faculty-id"):
            px1, py1, px2, py2 = parent["box"]
            p_height = max(py2 - py1, 1)
            y_rel = (iy - py1) / p_height

            # Chest area
            if 0.15 <= y_rel <= 0.60:
                item["confidence"] = min(0.99, conf * 1.50)
                
                iw = item["box"][2] - item["box"][0]
                ih = item["box"][3] - item["box"][1]
                aspect = ih / max(iw, 1)
                
                # Check for lanyard (vertical strip) or small rectangle
                if 0.3 <= aspect <= 5.0:
                    # Mark as weak signal if below normal threshold but matches chest geometry
                    if conf < self.CLASS_THRESHOLDS.get(label, 0.10):
                        item["label"] = f"{label}-weak"
                    return item
            else:
                if conf < 0.45:
                    return None

            if item["confidence"] < self.CLASS_THRESHOLDS.get(label, 0.10):
                return None
            return item

        if conf >= self.CLASS_THRESHOLDS.get(label, self.default_conf):
            return item

        return None

    def _parse_yolo(self, results) -> List[Dict[str, Any]]:
        """Parse ultralytics results into dicts."""
        processed = []
        for result in results:
            names = result.names
            if result.boxes is None:
                continue
            for box in result.boxes:
                label = names.get(int(box.cls[0]), "unknown")
                processed.append({
                    "label": label,
                    "confidence": round(float(box.conf[0]), 3),
                    "box": tuple(map(int, box.xyxy[0].tolist())),
                })
        return processed

    def draw_boxes(self, frame: np.ndarray, detections: List[Dict[str, Any]]) -> np.ndarray:
        """Draw colour-coded bounding boxes."""
        out = frame.copy()
        palette = {
            "person": (255, 255, 255), "blazer": (255, 140, 0),
            "formals": (255, 140, 0), "faculty-casuals": (0, 200, 255),
            "id-card": (0, 255, 255), "faculty-id": (0, 255, 150),
        }
        for det in detections:
            x1, y1, x2, y2 = det["box"]
            color = palette.get(det["label"], (60, 60, 255))
            cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
            label_str = f"{det['label']} {det['confidence']:.0%}"
            cv2.putText(out, label_str, (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        return out
