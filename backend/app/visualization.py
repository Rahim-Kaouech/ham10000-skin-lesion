"""Rendu de la carte Grad-CAM sans dépendance lourde (ni OpenCV, ni matplotlib)."""

from __future__ import annotations

import base64
import io

import numpy as np
from PIL import Image


def jet_colormap(x: np.ndarray) -> np.ndarray:
    """Approximation de la palette `jet` : x dans [0, 1] -> RGB uint8."""
    x = np.clip(x, 0.0, 1.0)
    r = np.clip(1.5 - np.abs(4 * x - 3), 0, 1)
    g = np.clip(1.5 - np.abs(4 * x - 2), 0, 1)
    b = np.clip(1.5 - np.abs(4 * x - 1), 0, 1)
    return (np.stack([r, g, b], axis=-1) * 255).astype(np.uint8)


def resize_cam(cam: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    img = Image.fromarray((np.clip(cam, 0, 1) * 255).astype(np.uint8))
    return np.asarray(img.resize(size, Image.BILINEAR), dtype=np.float32) / 255.0


def heatmap_rgba(cam: np.ndarray, size: tuple[int, int]) -> Image.Image:
    """Carte colorée dont l'opacité suit l'intensité (zones froides transparentes)."""
    c = resize_cam(cam, size)
    rgb = jet_colormap(c)
    alpha = (np.clip(c * 1.4, 0, 1) * 255).astype(np.uint8)
    return Image.fromarray(np.dstack([rgb, alpha]), mode="RGBA")


def overlay(image: Image.Image, cam: np.ndarray, alpha: float = 0.45) -> Image.Image:
    base = np.asarray(image.convert("RGB"), dtype=np.float32)
    heat = jet_colormap(resize_cam(cam, image.size)).astype(np.float32)
    out = (1 - alpha) * base + alpha * heat
    return Image.fromarray(np.clip(out, 0, 255).astype(np.uint8))


def to_png_bytes(image: Image.Image) -> bytes:
    buf = io.BytesIO()
    image.save(buf, format="PNG", optimize=True)
    return buf.getvalue()


def to_base64_png(image: Image.Image) -> str:
    return base64.b64encode(to_png_bytes(image)).decode("ascii")
