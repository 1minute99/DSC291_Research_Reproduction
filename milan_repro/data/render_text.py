"""Render a class-name text overlay onto an image.

Used to construct the synthetic spurious-text dataset from MILAN Section 7.
We draw the class name in the top-left corner over a translucent dark
stripe, mimicking the visual style of Fig 7 in the paper.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence, Tuple, Union

from PIL import Image, ImageDraw, ImageFont

Color = Tuple[int, int, int]


@dataclass
class TextStyle:
    font_size: int = 24
    font_color: Color = (255, 255, 255)
    stripe_color: Color = (0, 0, 0)
    stripe_alpha: int = 160
    margin_px: int = 4
    position: str = "top_left"   # only top_left supported for now
    font_path: Union[str, None] = None  # path to .ttf, else default bitmap font


_FONT_CANDIDATES: Sequence[str] = (
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
    "/Library/Fonts/Arial Bold.ttf",
)


def _load_font(style: TextStyle) -> ImageFont.ImageFont:
    candidates = [style.font_path] if style.font_path else list(_FONT_CANDIDATES)
    for path in candidates:
        if path and Path(path).exists():
            return ImageFont.truetype(path, style.font_size)
    # Fall back to PIL's built-in bitmap font (small but always available).
    return ImageFont.load_default()


def render(image: Image.Image, text: str, style: TextStyle = TextStyle()) -> Image.Image:
    """Return a copy of `image` with `text` rendered in the corner."""
    out = image.convert("RGB").copy()
    font = _load_font(style)

    # Measure text using getbbox (Pillow >=9) or getsize (older).
    if hasattr(font, "getbbox"):
        l, t, r, b = font.getbbox(text)
        text_w, text_h = r - l, b - t
    else:                                   # pragma: no cover
        text_w, text_h = font.getsize(text)

    pad = style.margin_px
    stripe_w = text_w + 2 * pad
    stripe_h = text_h + 2 * pad

    overlay = Image.new("RGBA", out.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    if style.position != "top_left":        # pragma: no cover
        raise NotImplementedError(style.position)

    x0, y0 = 0, 0
    x1, y1 = stripe_w, stripe_h
    draw.rectangle([x0, y0, x1, y1], fill=(*style.stripe_color, style.stripe_alpha))
    draw.text((pad, pad), text, font=font, fill=(*style.font_color, 255))

    out = Image.alpha_composite(out.convert("RGBA"), overlay).convert("RGB")
    return out
