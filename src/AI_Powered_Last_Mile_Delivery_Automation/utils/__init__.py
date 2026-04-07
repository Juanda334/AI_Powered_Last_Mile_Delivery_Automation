"""
AI_Powered_Last_Mile_Delivery_Automation.utils — shared helper functions.
"""
import os
import yaml
import json
import joblib
import logging
from pathlib import Path
from typing import Any

from ensure import ensure_annotations
from box import ConfigBox

logger = logging.getLogger("AI_Powered_Last_Mile_Delivery_Automation")


@ensure_annotations
def read_yaml(path_to_yaml: Path) -> ConfigBox:
    """Read a YAML file and return a ConfigBox for dot-access."""
    with open(path_to_yaml) as f:
        content = yaml.safe_load(f)
    logger.info(f"YAML loaded: {path_to_yaml}")
    return ConfigBox(content)


@ensure_annotations
def save_json(path: Path, data: dict) -> None:
    """Save a dictionary as a JSON file."""
    with open(path, "w") as f:
        json.dump(data, f, indent=4)
    logger.info(f"JSON saved: {path}")


@ensure_annotations
def load_json(path: Path) -> ConfigBox:
    """Load a JSON file and return a ConfigBox."""
    with open(path) as f:
        content = json.load(f)
    logger.info(f"JSON loaded: {path}")
    return ConfigBox(content)


@ensure_annotations
def save_model(path: Path, model: Any) -> None:
    """Persist a model with joblib."""
    joblib.dump(model, path)
    logger.info(f"Model saved: {path}")


@ensure_annotations
def load_model(path: Path) -> Any:
    """Load a joblib-persisted model."""
    model = joblib.load(path)
    logger.info(f"Model loaded: {path}")
    return model


@ensure_annotations
def create_directories(dirs: list, verbose: bool = True) -> None:
    """Create a list of directories if they don't exist."""
    for d in dirs:
        os.makedirs(d, exist_ok=True)
        if verbose:
            logger.info(f"Directory created: {d}")
