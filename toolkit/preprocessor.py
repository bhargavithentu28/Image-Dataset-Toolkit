"""
toolkit/preprocessor.py
───────────────────────
Resize images to a target resolution using three modes:

  letterbox   – aspect-ratio preserving resize with constant-colour padding
  stretch     – distorted resize to exact target dimensions
  crop_center – resize so the shorter side fills the target, then centre-crop

Original files are *never* modified.
Bounding-box transformations are applied when annotation records are passed.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
from PIL import Image  # noqa: F401 – available for callers

logger = logging.getLogger(__name__)

INTERPOLATION_MAP: dict[str, int] = {
    "INTER_NEAREST": cv2.INTER_NEAREST,
    "INTER_LINEAR": cv2.INTER_LINEAR,
    "INTER_CUBIC": cv2.INTER_CUBIC,
    "INTER_AREA": cv2.INTER_AREA,
    "INTER_LANCZOS4": cv2.INTER_LANCZOS4,
}


# ── Data model ────────────────────────────────────────────────────────────────

@dataclass
class LetterboxTransform:
    """Describes the letterbox transform applied to one image."""
    scale: float
    pad_top: int
    pad_left: int
    target_w: int
    target_h: int
    orig_w: int
    orig_h: int

    def transform_yolo_box(
        self, cx: float, cy: float, bw: float, bh: float
    ) -> tuple[float, float, float, float]:
        """
        Map a YOLO-normalised bounding box from original-image space to
        letterboxed-image space.

        Raises
        ------
        ValueError
            If the transformed box extends outside [0, 1].
        """
        abs_cx = cx * self.orig_w
        abs_cy = cy * self.orig_h
        abs_bw = bw * self.orig_w
        abs_bh = bh * self.orig_h

        new_cx = (abs_cx * self.scale + self.pad_left) / self.target_w
        new_cy = (abs_cy * self.scale + self.pad_top) / self.target_h
        new_bw = (abs_bw * self.scale) / self.target_w
        new_bh = (abs_bh * self.scale) / self.target_h

        for v, name in (
            (new_cx, "cx"), (new_cy, "cy"), (new_bw, "bw"), (new_bh, "bh")
        ):
            if not (0.0 <= v <= 1.0):
                raise ValueError(
                    f"Transformed bounding-box coordinate '{name}'={v:.4f} is out of "
                    f"[0, 1] after letterbox transform. Original box: "
                    f"cx={cx}, cy={cy}, bw={bw}, bh={bh}"
                )
        return new_cx, new_cy, new_bw, new_bh


@dataclass
class PreprocessResult:
    source_path: Path
    output_path: Path | None
    orig_width: int
    orig_height: int
    out_width: int | None = None
    out_height: int | None = None
    success: bool = False
    error: str | None = None
    transform: LetterboxTransform | None = None


@dataclass
class PreprocessSummary:
    results: list[PreprocessResult] = field(default_factory=list)

    @property
    def success_count(self) -> int:
        return sum(1 for r in self.results if r.success)

    @property
    def failure_count(self) -> int:
        return sum(1 for r in self.results if not r.success)


# ── Resize implementations ────────────────────────────────────────────────────

def _choose_interpolation(orig_w: int, orig_h: int, target_w: int, target_h: int,
                           up_interp: int, down_interp: int) -> int:
    if orig_w < target_w or orig_h < target_h:
        return up_interp
    return down_interp


def _letterbox(
    img: np.ndarray,
    target_w: int,
    target_h: int,
    pad_color: tuple[int, int, int],
    up_interp: int,
    down_interp: int,
) -> tuple[np.ndarray, LetterboxTransform]:
    orig_h, orig_w = img.shape[:2]
    scale = min(target_w / orig_w, target_h / orig_h)
    new_w = round(orig_w * scale)
    new_h = round(orig_h * scale)

    interp = _choose_interpolation(orig_w, orig_h, target_w, target_h, up_interp, down_interp)
    resized = cv2.resize(img, (new_w, new_h), interpolation=interp)

    pad_top = (target_h - new_h) // 2
    pad_left = (target_w - new_w) // 2
    pad_bottom = target_h - new_h - pad_top
    pad_right = target_w - new_w - pad_left

    channels = img.shape[2] if img.ndim == 3 else 1
    if channels == 4:
        pad_val = (*pad_color, 255)
    elif channels == 1:
        pad_val = (int(0.299 * pad_color[0] + 0.587 * pad_color[1] + 0.114 * pad_color[2]),)
    else:
        pad_val = pad_color

    padded = cv2.copyMakeBorder(
        resized, pad_top, pad_bottom, pad_left, pad_right,
        cv2.BORDER_CONSTANT, value=pad_val,
    )
    transform = LetterboxTransform(
        scale=scale, pad_top=pad_top, pad_left=pad_left,
        target_w=target_w, target_h=target_h, orig_w=orig_w, orig_h=orig_h,
    )
    return padded, transform


def _stretch(
    img: np.ndarray, target_w: int, target_h: int,
    up_interp: int, down_interp: int,
) -> np.ndarray:
    orig_h, orig_w = img.shape[:2]
    interp = _choose_interpolation(orig_w, orig_h, target_w, target_h, up_interp, down_interp)
    return cv2.resize(img, (target_w, target_h), interpolation=interp)


def _crop_center(
    img: np.ndarray, target_w: int, target_h: int,
    up_interp: int, down_interp: int,
) -> np.ndarray:
    orig_h, orig_w = img.shape[:2]
    scale = max(target_w / orig_w, target_h / orig_h)
    new_w = round(orig_w * scale)
    new_h = round(orig_h * scale)
    interp = _choose_interpolation(orig_w, orig_h, target_w, target_h, up_interp, down_interp)
    resized = cv2.resize(img, (new_w, new_h), interpolation=interp)
    x0 = (new_w - target_w) // 2
    y0 = (new_h - target_h) // 2
    return resized[y0: y0 + target_h, x0: x0 + target_w]


# ── Public API ────────────────────────────────────────────────────────────────

def preprocess_images(
    records,
    output_dir: str | Path,
    target_width: int = 640,
    target_height: int = 640,
    mode: str = "letterbox",
    pad_color: tuple[int, int, int] = (114, 114, 114),
    interpolation_upscale: str = "INTER_CUBIC",
    interpolation_downscale: str = "INTER_AREA",
    preserve_alpha: bool = True,
) -> PreprocessSummary:
    """
    Preprocess all valid records and write to *output_dir*.

    Parameters
    ----------
    records:
        Iterable of :class:`~toolkit.scanner.ImageRecord`.
    output_dir:
        Destination for processed images.
    target_width, target_height:
        Output resolution.
    mode:
        ``"letterbox"`` | ``"stretch"`` | ``"crop_center"``.
    pad_color:
        RGB pad colour for letterbox mode.
    interpolation_upscale, interpolation_downscale:
        OpenCV interpolation flag names.
    preserve_alpha:
        Keep alpha channel when present.

    Returns
    -------
    PreprocessSummary
    """
    if mode not in ("letterbox", "stretch", "crop_center"):
        raise ValueError(f"Unknown mode '{mode}'. Choose letterbox / stretch / crop_center.")

    up_interp = INTERPOLATION_MAP.get(interpolation_upscale, cv2.INTER_CUBIC)
    down_interp = INTERPOLATION_MAP.get(interpolation_downscale, cv2.INTER_AREA)
    out_root = Path(output_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    summary = PreprocessSummary()
    valid = [r for r in records if r.is_valid]
    logger.info("Preprocessing %d images → %s (mode=%s)", len(valid), out_root, mode)

    for rec in valid:
        out_path = out_root / rec.relative_path
        out_path.parent.mkdir(parents=True, exist_ok=True)

        img = cv2.imread(
            str(rec.path),
            cv2.IMREAD_UNCHANGED if preserve_alpha else cv2.IMREAD_COLOR,
        )
        if img is None:
            summary.results.append(PreprocessResult(
                source_path=rec.path, output_path=None,
                orig_width=rec.width or 0, orig_height=rec.height or 0,
                success=False, error="OpenCV could not decode image",
            ))
            continue

        orig_h, orig_w = img.shape[:2]
        transform: LetterboxTransform | None = None

        try:
            if mode == "letterbox":
                processed, transform = _letterbox(
                    img, target_width, target_height, pad_color, up_interp, down_interp
                )
            elif mode == "stretch":
                processed = _stretch(img, target_width, target_height, up_interp, down_interp)
            else:
                processed = _crop_center(img, target_width, target_height, up_interp, down_interp)

            cv2.imwrite(str(out_path), processed)
            out_h, out_w = processed.shape[:2]

            summary.results.append(PreprocessResult(
                source_path=rec.path, output_path=out_path,
                orig_width=orig_w, orig_height=orig_h,
                out_width=out_w, out_height=out_h,
                success=True, transform=transform,
            ))
            logger.debug("Preprocessed %s → %s", rec.relative_path, out_path)

        except Exception as exc:  # noqa: BLE001
            logger.error("Failed to preprocess %s: %s", rec.relative_path, exc)
            summary.results.append(PreprocessResult(
                source_path=rec.path, output_path=None,
                orig_width=orig_w, orig_height=orig_h,
                success=False, error=str(exc),
            ))

    logger.info(
        "Preprocessing done: %d success, %d failed",
        summary.success_count, summary.failure_count,
    )
    return summary
