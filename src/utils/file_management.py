from pathlib import Path
from typing import Any

import joblib


def save_data(data: Any, path: Path) -> None:
    """Save data to a file."""
    with path.open("wb") as f:
        joblib.dump(data, f)


def load_data(path: Path) -> Any:
    """Load data from a file."""
    with path.open("rb") as f:
        return joblib.load(f)