"""
dress_classifier.py
--------------------
IDVestAI — Dress Code Classifier Module

Classifies whether a person is wearing FORMAL or CASUAL attire
based on the detected object labels from YOLOv8.

Architecture Position:
    Camera → Detector → IDCardDetector → **DressCodeClassifier** → Logic Engine

Rule:
    if "formal_shirt" OR "tie" is detected → "formal"
    else                                    → "casual"

Usage:
    from dress_classifier import DressCodeClassifier

    clf = DressCodeClassifier()
    result = clf.classify(detections)   # "formal" or "casual"
    print(result)
"""

from __future__ import annotations

from typing import List, Dict, Any


# ─────────────────────────────────────────────────────────────────────────────
# DressCodeClassifier class
# ─────────────────────────────────────────────────────────────────────────────

class DressCodeClassifier:
    """
    Rule-based dress code classifier.

    Rules (beginner-friendly, easy to extend):
      - "formal_shirt" detected  →  formal
      - "tie" detected           →  formal
      - Neither detected         →  casual
    """

    # Labels that count as formal attire
    FORMAL_LABELS = {"formals", "blazer"}

    def classify(self, detections: List[Dict[str, Any]]) -> str:
        """
        Classify attire as 'formal' or 'casual' based on detections.

        Parameters
        ----------
        detections : list — output from Detector.detect()

        Returns
        -------
        str — "formal" if formal attire detected, else "casual"
        """
        detected_labels = {d.get("label") for d in detections}

        # Check if any formal label is in the detected labels
        if detected_labels & self.FORMAL_LABELS:   # set intersection
            return "formal"
        return "casual"

    def get_formal_items(self, detections: List[Dict[str, Any]]) -> List[str]:
        """
        Return the list of formal attire items found in the frame.

        
        """
        found = []
        for det in detections:
            label = det.get("label", "")
            if label in self.FORMAL_LABELS and label not in found:
                found.append(label)
        return found
