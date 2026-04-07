"""
logic.py
--------
IDVest AI — Standalone Logic Engine
"""

from __future__ import annotations

import time
import datetime
from typing import List, Dict, Any, Tuple, Set

class LogicEngine:
    LBL_PERSON          = "person"
    LBL_BLAZER          = "blazer"
    LBL_FORMALS         = "formals"
    LBL_FACULTY_CASUALS = "faculty-casuals"
    LBL_ID_CARD         = "id-card"
    LBL_FACULTY_ID      = "faculty-id"

    def __init__(self):
        self.tracks = {}  # tid -> (cx, cy, last_seen)
        self.history = {} # tid -> {"id_status": [...], "attire": [...]}
        self.next_id = 1

    def _assign_ids(self, persons: List[Dict[str, Any]], now: float) -> List[str]:
        # Expire old tracks
        for tid, (tcx, tcy, tlast) in list(self.tracks.items()):
            if now - tlast > 5.0:
                del self.tracks[tid]
                if tid in self.history:
                    del self.history[tid]

        assigned_ids = [None] * len(persons)
        distances = []
        for i, p in enumerate(persons):
            x1, y1, x2, y2 = p["box"]
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            for tid, (tcx, tcy, _) in self.tracks.items():
                dist = ((cx - tcx)**2 + (cy - tcy)**2)**0.5
                distances.append((dist, i, tid))
        
        distances.sort()
        used_persons = set()
        used_tracks = set()

        for dist, i, tid in distances:
            if dist < 50 and i not in used_persons and tid not in used_tracks:
                assigned_ids[i] = tid
                used_persons.add(i)
                used_tracks.add(tid)

        for i, p in enumerate(persons):
            if assigned_ids[i] is None:
                tid = f"P{self.next_id}"
                self.next_id += 1
                assigned_ids[i] = tid
            
            x1, y1, x2, y2 = p["box"]
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            self.tracks[assigned_ids[i]] = (cx, cy, now)
        
        return assigned_ids

    def evaluate(self, detections: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        now = time.time()
        is_monday = datetime.date.today().weekday() == 0

        persons = [d for d in detections if d["label"] == self.LBL_PERSON]
        items   = [d for d in detections if d["label"] != self.LBL_PERSON]

        results = []
        person_ids = self._assign_ids(persons, now)

        for p, person_id in zip(persons, person_ids):
            box = p["box"]
            my_classes = self._get_associated_classes(box, items)
            verdict = self._evaluate_person(person_id, box, my_classes, is_monday)
            results.append(verdict)

        return results

    def get_frame_verdict(self, person_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Aggregate results for the frame (used by api_server)."""
        if not person_results:
            return {
                "status": "No Person Detected",
                "is_compliant": True,
                "has_id": False,
                "dress_code": "unknown",
                "formal_items": [],
                "person_results": []
            }
        
        # Consider frame compliant if all processed persons are compliant
        is_compliant = all(r["compliance"] == "Compliant" for r in person_results)
        
        first = person_results[0]
        has_id = first["id_status"] == "Present"
        dress = first["attire"]
        
        if is_compliant:
            status = "Compliant"
        elif not has_id:
            status = "Missing ID"
        else:
            status = "Improper Dress"

        return {
            "status": status,
            "is_compliant": is_compliant,
            "has_id": has_id,
            "dress_code": dress.lower(),
            "formal_items": [],
            "person_results": person_results
        }

    def _evaluate_person(
        self, person_id: str, box: List[int], classes: Set[str], is_monday: bool
    ) -> Dict[str, Any]:
        alerts = []
        is_faculty = self.LBL_FACULTY_ID in classes

        # Base detections
        has_blazer = self.LBL_BLAZER in classes or self.LBL_FORMALS in classes
        has_id = self.LBL_ID_CARD in classes or self.LBL_FACULTY_ID in classes
        is_weak_id = f"{self.LBL_ID_CARD}-weak" in classes or f"{self.LBL_FACULTY_ID}-weak" in classes
        
        # Combine rules: blazer + weak chest object (lanyard/card signal) -> assume ID present
        if has_blazer and is_weak_id:
            has_id = True

        # History buffering logic
        if person_id not in self.history:
            self.history[person_id] = {"id_status": [], "attire": []}
        
        hist = self.history[person_id]
        hist["id_status"].append(has_id)
        hist["attire"].append(has_blazer)

        if len(hist["id_status"]) > 5:
            hist["id_status"].pop(0)
        if len(hist["attire"]) > 5:
            hist["attire"].pop(0)

        # Majority voting (3 out of 5 frames required for stability)
        vote_id = sum(1 for val in hist["id_status"] if val)
        vote_blazer = sum(1 for val in hist["attire"] if val)

        stable_has_id = vote_id >= min(3, len(hist["id_status"]))
        stable_has_blazer = vote_blazer >= min(3, len(hist["attire"]))

        # Formatting Output variables
        id_str = "Present" if stable_has_id else "Missing"
        attire_str = "Formal" if stable_has_blazer else "Not Formal"

        # Requested logical evaluation schema
        if stable_has_blazer and stable_has_id:
            compliance = "Compliant"
        elif stable_has_blazer and not stable_has_id:
            compliance = "Violation"
            alerts.append("Missing ID")
        else:
            compliance = "Violation"
            alerts.append("Improper Dress")

        return {
            "person_id": person_id,
            "attire": attire_str,
            "id_status": id_str,
            "compliance": compliance,
            "box": box,
            "alerts": alerts
        }

    def _get_associated_classes(
        self, person_box: List[int], items: List[Dict[str, Any]]
    ) -> Set[str]:
        px1, py1, px2, py2 = person_box
        margin = 20
        classes = set()
        for item in items:
            ix = (item["box"][0] + item["box"][2]) / 2
            iy = (item["box"][1] + item["box"][3]) / 2
            if (px1 - margin <= ix <= px2 + margin) and (py1 - margin <= iy <= py2 + margin):
                classes.add(item["label"])
        return classes

