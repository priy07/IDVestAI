"""
alert.py
--------
IDVestAI — Alert System Module

Handles two types of alerts:
  1. Console / print alerts  — always active, shows in terminal
  2. JSON log alerts         — writes every event to alerts/violations.json
"""

from __future__ import annotations

import json
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Set


class AlertSystem:
    """
    Manages real-time dress-code alerts and prevents spam.

    Features:
    - Maintains `alerted_ids` set to ensure ONE alert per person.
    """

    def __init__(
        self,
        log_dir: str = "alerts",
    ) -> None:
        self._log_dir          = Path(log_dir)
        self._log_file         = self._log_dir / "violations.json"
        
        # Rate-limiting / anti-spam
        self.alerted_ids: Set[str] = set()

        # Create alerts directory if it doesn't exist
        self._log_dir.mkdir(parents=True, exist_ok=True)

        if not self._log_file.exists():
            self._log_file.write_text("[]", encoding="utf-8")

        print(f"[AlertSystem] Log file: {self._log_file.resolve()}")

    def trigger(self, camera_id: str, verdict: Dict[str, Any], image_path: str = None) -> None:
        """
        Process a detection verdict and fire the appropriate alert once per person.
        """
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        person_results = verdict.get("person_results", [])
        
        for person in person_results:
            person_id = person["person_id"]
            compliance = person["compliance"]
            
            # If the person violates the rules and hasn't been alerted yet
            if compliance == "Violation" and person_id not in self.alerted_ids:
                
                # Setup details
                attire = person["attire"]
                id_status = person["id_status"]
                
                img_str = f" | Image: {image_path}" if image_path else ""
                print(f"[{timestamp}] 🚨  VIOLATION  | Camera: {camera_id} | "
                      f"Person: {person_id} | ID: {id_status} | Dress: {attire}{img_str}")
                
                # Write to JSON
                self._write_to_json(camera_id, person, timestamp, image_path)
                
                # Mark as alerted to prevent spam
                self.alerted_ids.add(person_id)

    def _write_to_json(
        self,
        camera_id: str,
        person: Dict[str, Any],
        timestamp: str,
        image_path: str = None
    ) -> None:
        """Append a log entry to the JSON file for this person_id."""
        entry = {
            "timestamp":    timestamp,
            "camera_id":    camera_id,
            "person_id":    person["person_id"],
            "status":       "Violation",
            "has_id":       person["id_status"] == "Present",
            "dress_code":   person["attire"],
            "is_compliant": False,
        }
        if image_path:
            entry["image"] = image_path

        try:
            with open(self._log_file, "r", encoding="utf-8") as f:
                data = json.load(f)

            data.append(entry)

            with open(self._log_file, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)

        except (json.JSONDecodeError, IOError) as e:
            print(f"[AlertSystem] JSON write error: {e} — resetting log file.")
            with open(self._log_file, "w", encoding="utf-8") as f:
                json.dump([entry], f, indent=2)
