"""
id_detector.py
--------------
IDVestAI — ID Card Detection Module

Receives a list of detection results from detector.py and determines
whether an ID card is present in the frame.

Architecture Position:
    Camera → Detector → **IDCardDetector** → DressCodeClassifier → ...

Usage:
    from id_detector import IDCardDetector

    checker = IDCardDetector()
    has_id = checker.has_id_card(detections)   # True / False
"""

from __future__ import annotations

from typing import List, Dict, Any


# ─────────────────────────────────────────────────────────────────────────────
# IDCardDetector class
# ─────────────────────────────────────────────────────────────────────────────

class IDCardDetector:
    """
    Checks whether an 'id_card' label is present in the detection results.

    This is intentionally simple so beginners can easily understand the logic.
    """

    # The label name we look for — matches the class name in your YOLO model
    ID_CARD_LABELS = {"id-card", "faculty-id"}

    def has_id_card(self, detections):
        """
        Scan the list of detections and return True if at least one
        'id_card' object was detected with any confidence.

        Parameters
        ----------
        detections : list — output from Detector.detect()

        Returns
        -------
        bool — True if ID card found, False otherwise
        """
        for det in detections:
            if det.get("label") in self.ID_CARD_LABELS:
                return True
        return False

    def get_id_card_detections(self, detections):
        return [d for d in detections if d.get("label") in self.ID_CARD_LABELS]
