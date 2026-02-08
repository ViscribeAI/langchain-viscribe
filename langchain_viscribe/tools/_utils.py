"""Shared utilities for image tools."""

import base64
from pathlib import Path


def load_image_path_to_base64(image_path: str) -> str:
    """Load a local image file and return its base64-encoded string.

    Args:
        image_path: Absolute or relative path to the image file.

    Returns:
        Base64-encoded string of the image content (UTF-8 decoded).

    Raises:
        FileNotFoundError: If the path does not exist.
        ValueError: If the path is not a file.
    """
    path = Path(image_path).resolve()
    if not path.exists():
        raise FileNotFoundError(f"Image file not found: {path}")
    if not path.is_file():
        raise ValueError(f"Path is not a file: {path}")
    data = path.read_bytes()
    return base64.b64encode(data).decode("utf-8")
