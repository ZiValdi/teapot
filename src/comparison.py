"""Scenario storage and management for comparison."""

from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any

HISTORY_DIR = Path("history")

def ensure_history_dir():
    """Create history directory if it doesn't exist."""
    HISTORY_DIR.mkdir(exist_ok=True)

def save_scenario(scenario: dict[str, Any], custom_name: str | None = None) -> str:
    """
    Save a scenario to disk.
    If custom_name is provided, use it; otherwise use timestamp.
    Returns the filename saved.
    """
    ensure_history_dir()
    if custom_name:
        # Sanitize filename
        name = "".join(c for c in custom_name if c.isalnum() or c in " _-")
        if not name:
            name = "scenario"
        filename = f"{name}.json"
    else:
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        filename = f"scenario_{timestamp}.json"
    # Avoid overwriting
    counter = 1
    base = filename
    while (HISTORY_DIR / filename).exists():
        name_part = base.rsplit(".json", 1)[0]
        filename = f"{name_part}_{counter}.json"
        counter += 1
    with open(HISTORY_DIR / filename, "w") as f:
        json.dump(scenario, f, indent=2, default=str)
    return filename

def load_scenarios() -> list[dict[str, Any]]:
    """Load all scenarios from the history folder."""
    ensure_history_dir()
    scenarios = []
    for file in HISTORY_DIR.glob("*.json"):
        try:
            with open(file, "r") as f:
                data = json.load(f)
                data["_file"] = file.name  # store filename for deletion
                scenarios.append(data)
        except Exception:
            continue
    # Sort by timestamp (newest first)
    scenarios.sort(key=lambda x: x.get("timestamp", ""), reverse=True)
    return scenarios

def delete_scenario(filename: str) -> bool:
    """Delete a scenario file."""
    try:
        (HISTORY_DIR / filename).unlink()
        return True
    except Exception:
        return False