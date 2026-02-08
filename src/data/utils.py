"""Image and data utility functions."""

from __future__ import annotations

import base64
import io
from pathlib import Path

from PIL import Image


def load_image(image_source: str | Path | Image.Image) -> Image.Image:
    """Load an image from a file path, URL, or PIL Image.

    Args:
        image_source: Path to image file, or PIL Image object.

    Returns:
        PIL Image in RGB mode.
    """
    if isinstance(image_source, Image.Image):
        return image_source.convert("RGB")
    path = Path(image_source)
    if path.exists():
        return Image.open(path).convert("RGB")
    raise ValueError(f"Cannot load image from: {image_source}")


def resize_image(image: Image.Image, max_size: int = 1280) -> Image.Image:
    """Resize image so the longest side is at most max_size pixels.

    Args:
        image: PIL Image to resize.
        max_size: Maximum dimension in pixels.

    Returns:
        Resized PIL Image.
    """
    w, h = image.size
    if max(w, h) <= max_size:
        return image
    scale = max_size / max(w, h)
    new_w = int(w * scale)
    new_h = int(h * scale)
    return image.resize((new_w, new_h), Image.LANCZOS)


def image_to_base64(image: Image.Image, fmt: str = "PNG") -> str:
    """Convert PIL Image to base64 string.

    Args:
        image: PIL Image.
        fmt: Image format (PNG, JPEG).

    Returns:
        Base64-encoded string.
    """
    buffer = io.BytesIO()
    image.save(buffer, format=fmt)
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def normalize_text(text: str) -> str:
    """Normalize text for comparison: lowercase, strip whitespace, collapse spaces."""
    text = text.lower().strip()
    # Collapse multiple whitespace characters into single space
    return " ".join(text.split())
