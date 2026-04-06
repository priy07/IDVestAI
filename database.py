"""
database.py
-----------
IDVestAI — Database / Logging Module

Stores detection records in a JSON file that acts as a simple
"database" — perfect for academic projects and demos.

Architecture Position:
    AlertSystem → **Database** ← AdminDashboard (reads)

Schema (each record):
    {
        "id":           int,           # auto-increment primary key
        "timestamp":    str,           # ISO 8601 datetime
        "camera_id":    str,           # e.g. "CAM-01"
        "id_status":    bool,          # True = ID card present
        "dress_status": str,           # "formal" | "casual" | "unknown"
        "violation":    str,           # violation type or "None"
        "is_compliant": bool           # True if fully compliant
    }

Usage:
    from database import Database

    db = Database()
    db.save(camera_id="CAM-01", verdict=verdict_dict)
    records = db.get_all()
    summary = db.get_summary()
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional


# ─────────────────────────────────────────────────────────────────────────────
# Database class
# ─────────────────────────────────────────────────────────────────────────────

class Database:
    """
    Simple JSON-file-backed database for IDVestAI detection logs.

    In production you would replace this with SQLite / PostgreSQL,
    but JSON is perfect for academic demos — no setup required.
    """

    def __init__(self, db_path: str = "logs/detections.json") -> None:
        self._path = Path(db_path)
        self._path.parent.mkdir(parents=True, exist_ok=True)

        # Create file with empty list if it doesn't exist
        if not self._path.exists():
            self._write([])
            print(f"[Database] Created new database: {self._path.resolve()}")
        else:
            print(f"[Database] Using existing database: {self._path.resolve()}")

    # ── Public API ────────────────────────────────────────────────────────────

    def save(self, camera_id: str, verdict: Dict[str, Any], image_path: str = None) -> int:
        """
        Persist a detection result as a new record in the database.

        Parameters
        ----------
        camera_id : str  — camera label (e.g. "CAM-01")
        verdict   : dict — output from LogicEngine.evaluate()

        Returns
        -------
        int — the auto-generated record ID
        """
        records = self._read()

        # Auto-increment ID
        new_id = (records[-1]["id"] + 1) if records else 1

        # Map violation type
        status = verdict.get("status", "Unknown")
        violation = (
            "None" if verdict.get("is_compliant") else status
        )

        record = {
            "id":           new_id,
            "timestamp":    datetime.now().isoformat(timespec="seconds"),
            "camera_id":    camera_id,
            "id_status":    bool(verdict.get("has_id", False)),
            "dress_status": verdict.get("dress_code", "unknown"),
            "violation":    violation,
            "is_compliant": bool(verdict.get("is_compliant", False)),
        }
        if image_path:
            record["image"] = image_path

        records.append(record)
        self._write(records)
        return new_id

    def get_all(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Retrieve all records, newest first.

        Parameters
        ----------
        limit : int or None — if set, return only the last N records

        Returns
        -------
        list of dicts
        """
        records = list(reversed(self._read()))  # newest first
        if limit is not None:
            return records[:limit]
        return records

    def get_summary(self) -> Dict[str, Any]:
        """
        Return aggregate statistics for the Admin Dashboard.

        Returns
        -------
        dict with keys:
            total_detections : int
            total_violations : int
            total_compliant  : int
            compliance_rate  : str  (e.g. "73.5%")
            violation_types  : dict (e.g. {"Missing ID": 5, "Improper Dress": 3})
        """
        records = self._read()
        total = len(records)

        if total == 0:
            return {
                "total_detections": 0,
                "total_violations": 0,
                "total_compliant":  0,
                "compliance_rate":  "N/A",
                "violation_types":  {},
            }

        compliant  = sum(1 for r in records if r.get("is_compliant"))
        violations = total - compliant

        # Count violation types
        violation_types: Dict[str, int] = {}
        for r in records:
            v = r.get("violation", "None")
            if v != "None":
                violation_types[v] = violation_types.get(v, 0) + 1

        rate = f"{(compliant / total * 100):.1f}%"

        return {
            "total_detections": total,
            "total_violations": violations,
            "total_compliant":  compliant,
            "compliance_rate":  rate,
            "violation_types":  violation_types,
        }

    def clear(self) -> None:
        """Delete all records (useful for testing/demo resets)."""
        self._write([])
        print("[Database] All records cleared.")

    # ── Private helpers ───────────────────────────────────────────────────────

    def _read(self) -> List[Dict[str, Any]]:
        """Load and return all records from disk."""
        try:
            with open(self._path, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            return []

    def _write(self, records: List[Dict[str, Any]]) -> None:
        """Write records list to disk."""
        with open(self._path, "w", encoding="utf-8") as f:
            json.dump(records, f, indent=2, ensure_ascii=False)
