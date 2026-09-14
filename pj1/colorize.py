"""Colorize vertically stacked Prokudin-Gorskii glass-plate scans.

The input order is blue, green, red from top to bottom.  A reported shift
``(x, y)`` is the translation applied to a moving channel to align it with
the blue reference channel: positive x moves right and positive y moves down.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, Literal

import numpy as np
from PIL import Image


Metric = Literal["ncc", "l2", "edge-ncc"]
Shift = tuple[int, int]


def read_grayscale(path: Path) -> np.ndarray:
    """Read a plate scan as a float32 grayscale image in [0, 1]."""
    with Image.open(path) as image:
        array = np.asarray(image)

    if array.ndim == 3:
        # This is mainly a convenience for accidentally RGB-encoded scans.
        # to prevent RGBA; mean takes the mean of RGB if the input is 3-dim
        array = array[..., :3].astype(np.float32).mean(axis=2)

    if np.issubdtype(array.dtype, np.integer):
        scale = float(np.iinfo(array.dtype).max)
    else:
        scale = float(np.nanmax(array)) if np.nanmax(array) > 1 else 1.0
    return np.clip(array.astype(np.float32) / scale, 0.0, 1.0)


def split_channels(plate: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Split a stacked plate into equally sized B, G, and R channels."""
    if plate.ndim != 2:
        raise ValueError(f"Expected a 2-D grayscale image, got shape {plate.shape}")
    height = plate.shape[0] // 3
    if height == 0:
        raise ValueError("The input is too short to contain three channels")
    # Any one- or two-row remainder is intentionally discarded.
    return plate[:height], plate[height : 2 * height], plate[2 * height : 3 * height]


def _edge_magnitude(image: np.ndarray) -> np.ndarray:
    """Return a simple finite-difference edge magnitude."""
    dy, dx = np.gradient(image) # brightness variation along x and y
    return np.hypot(dx, dy).astype(np.float32, copy=False) # we don't care about direction, so hypot


def _downsample(image: np.ndarray) -> np.ndarray:
    """Gaussian prefilter and decimate by two (one pyramid step)."""
    # Separable binomial kernel [1, 4, 6, 4, 1] / 16, prevent aliasing by Guassian.
    # inventing a reflection frame around the scan before subsampling.
    padded_x = np.pad(image, ((0, 0), (2, 2)), mode="reflect")
    blurred_x = (
        padded_x[:, :-4]
        + 4 * padded_x[:, 1:-3]
        + 6 * padded_x[:, 2:-2]
        + 4 * padded_x[:, 3:-1]
        + padded_x[:, 4:]
    ) / 16.0
    padded_y = np.pad(blurred_x, ((2, 2), (0, 0)), mode="reflect")
    blurred = (
        padded_y[:-4]
        + 4 * padded_y[1:-3]
        + 6 * padded_y[2:-2]
        + 4 * padded_y[3:-1]
        + padded_y[4:]
    ) / 16.0
    return np.ascontiguousarray(blurred[::2, ::2], dtype=np.float32)


def _overlap(
    reference: np.ndarray, moving: np.ndarray, shift: Shift, crop_fraction: float
) -> tuple[np.ndarray, np.ndarray]:
    """Get corresponding valid pixels after translating moving by (x, y).
    
    Examples
    --------
    shift = (3,5) stands for moving one needs to moved right 3 pixels and down 5 pixels
    Thus we take moving[0:97, 0:95] to compare with reference[3:100, 5:100]
    """
    dx, dy = shift
    height, width = reference.shape
    ref_y0, ref_y1 = max(0, dy), min(height, height + dy) # y range of reference BLUE channelS
    ref_x0, ref_x1 = max(0, dx), min(width, width + dx)
    mov_y0, mov_y1 = ref_y0 - dy, ref_y1 - dy
    mov_x0, mov_x1 = ref_x0 - dx, ref_x1 - dx

    # we only take the central (1-crop_fraction) part into score calculation
    overlap_h, overlap_w = ref_y1 - ref_y0, ref_x1 - ref_x0
    trim_y = int(overlap_h * crop_fraction)
    trim_x = int(overlap_w * crop_fraction)
    ref_y0, ref_y1 = ref_y0 + trim_y, ref_y1 - trim_y
    ref_x0, ref_x1 = ref_x0 + trim_x, ref_x1 - trim_x
    mov_y0, mov_y1 = mov_y0 + trim_y, mov_y1 - trim_y
    mov_x0, mov_x1 = mov_x0 + trim_x, mov_x1 - trim_x

    return (
        reference[ref_y0:ref_y1, ref_x0:ref_x1],
        moving[mov_y0:mov_y1, mov_x0:mov_x1],
    )


def _score(reference: np.ndarray, moving: np.ndarray, metric: Metric) -> float:
    """Return a score to minimize."""
    if reference.size == 0:
        return float("inf")
    ref = reference.ravel()
    mov = moving.ravel()
    if metric == "l2":
        difference = ref - mov
        return float(np.mean(difference * difference))

    ref_centered = ref - ref.mean()
    mov_centered = mov - mov.mean()
    denominator = float(np.linalg.norm(ref_centered) * np.linalg.norm(mov_centered))
    if denominator <= 1e-12:
        return float("inf")
    # NCC is maximized, so negate it to share the minimization path with L2.
    return -float(np.dot(ref_centered, mov_centered) / denominator)


def align_single_scale(
    moving: np.ndarray,
    reference: np.ndarray,
    radius: int = 15,
    metric: Metric = "ncc",
    center: Shift = (0, 0),
    crop_fraction: float = 0.10,
) -> tuple[Shift, float]:
    """Exhaustively search a square window around ``center``."""
    if moving.shape != reference.shape:
        raise ValueError("The moving and reference channels must have equal shapes")
    if radius < 0:
        raise ValueError("Search radius must be non-negative")
    if not 0 <= crop_fraction < 0.5:
        raise ValueError("Crop fraction must be in [0, 0.5)")

    score_metric: Metric = "ncc" if metric == "edge-ncc" else metric
    if metric == "edge-ncc":
        reference = _edge_magnitude(reference)
        moving = _edge_magnitude(moving)

    center_x, center_y = center
    best_shift = center
    best_score = float("inf")
    for dy in range(center_y - radius, center_y + radius + 1):
        for dx in range(center_x - radius, center_x + radius + 1):
            ref_patch, mov_patch = _overlap(
                reference, moving, (dx, dy), crop_fraction
            )
            score = _score(ref_patch, mov_patch, score_metric)
            if score < best_score:
                best_score = score
                best_shift = (dx, dy)
    return best_shift, best_score


def align_pyramid(
    moving: np.ndarray,
    reference: np.ndarray,
    metric: Metric = "edge-ncc",
    coarse_radius: int = 15,
    refine_radius: int = 3,
    min_size: int = 400,
    crop_fraction: float = 0.10,
) -> tuple[Shift, float]:
    """Align from the coarsest Gaussian-pyramid level to the original image."""
    moving_pyramid = [moving]
    reference_pyramid = [reference]
    while min(reference_pyramid[-1].shape) > min_size:
        moving_pyramid.append(_downsample(moving_pyramid[-1]))
        reference_pyramid.append(_downsample(reference_pyramid[-1]))

    estimate: Shift = (0, 0)
    best_score = float("inf")
    for level in range(len(reference_pyramid) - 1, -1, -1):
        if level != len(reference_pyramid) - 1:
            estimate = (estimate[0] * 2, estimate[1] * 2)
        radius = coarse_radius if level == len(reference_pyramid) - 1 else refine_radius
        estimate, best_score = align_single_scale(
            moving_pyramid[level],
            reference_pyramid[level],
            radius=radius,
            metric=metric,
            center=estimate,
            crop_fraction=crop_fraction,
        )
    return estimate, best_score


def compose_rgb(
    blue: np.ndarray,
    green: np.ndarray,
    red: np.ndarray,
    green_shift: Shift,
    red_shift: Shift,
    border_crop: float = 0.05,
) -> np.ndarray:
    """Translate channels, intersect valid regions, and stack them as RGB."""
    if not 0 <= border_crop < 0.5:
        raise ValueError("Border crop must be in [0, 0.5)")
    height, width = blue.shape
    shifts = ((0, 0), green_shift, red_shift)
    left = max(dx for dx, _ in shifts)
    right = min(width + dx for dx, _ in shifts)
    top = max(dy for _, dy in shifts)
    bottom = min(height + dy for _, dy in shifts)
    if left >= right or top >= bottom:
        raise ValueError("Estimated shifts leave no common image region")

    def valid_region(channel: np.ndarray, shift: Shift) -> np.ndarray:
        dx, dy = shift
        return channel[top - dy : bottom - dy, left - dx : right - dx]

    rgb = np.dstack(
        (valid_region(red, red_shift), valid_region(green, green_shift),
         valid_region(blue, (0, 0)))
    )
    trim_y = int(rgb.shape[0] * border_crop)
    trim_x = int(rgb.shape[1] * border_crop)
    if trim_y:
        rgb = rgb[trim_y:-trim_y]
    if trim_x:
        rgb = rgb[:, trim_x:-trim_x]
    return rgb


def save_rgb(image: np.ndarray, path: Path, autocontrast: bool = False) -> None:
    """Save a float RGB image, optionally stretching each channel robustly."""
    output = image.copy()
    if autocontrast:
        for channel_index in range(3):
            channel = output[..., channel_index]
            low, high = np.percentile(channel, (1, 99))
            if high > low:
                output[..., channel_index] = (channel - low) / (high - low)
    output_uint8 = np.clip(output * 255.0, 0, 255).astype(np.uint8)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(output_uint8).save(path, quality=95)


def colorize(
    input_path: Path,
    output_path: Path,
    method: Literal["single", "pyramid"] = "pyramid",
    metric: Metric = "edge-ncc",
    search_radius: int = 15,
    refine_radius: int = 3,
    min_size: int = 400,
    score_crop: float = 0.10,
    border_crop: float = 0.05,
    autocontrast: bool = False,
) -> tuple[Shift, Shift]:
    """Colorize one scan and return the G-to-B and R-to-B shifts."""
    plate = read_grayscale(input_path)
    blue, green, red = split_channels(plate)
    if method == "single":
        green_shift, _ = align_single_scale(
            green, blue, search_radius, metric, crop_fraction=score_crop
        )
        red_shift, _ = align_single_scale(
            red, blue, search_radius, metric, crop_fraction=score_crop
        )
    else:
        alignment_options = dict(
            metric=metric,
            coarse_radius=search_radius,
            refine_radius=refine_radius,
            min_size=min_size,
            crop_fraction=score_crop,
        )
        green_shift, _ = align_pyramid(green, blue, **alignment_options)
        red_shift, _ = align_pyramid(red, blue, **alignment_options)

    rgb = compose_rgb(
        blue, green, red, green_shift, red_shift, border_crop=border_crop
    )
    save_rgb(rgb, output_path, autocontrast=autocontrast)
    return green_shift, red_shift


def _default_output(input_path: Path, output_dir: Path) -> Path:
    return output_dir / f"{input_path.stem}_color.jpg"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Align and colorize vertically stacked BGR glass-plate scans."
    )
    parser.add_argument("inputs", type=Path, nargs="+", help="input .jpg/.tif scan(s)")
    parser.add_argument("-o", "--output", type=Path, help="output path (one input only)")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--method", choices=("single", "pyramid"), default="pyramid")
    parser.add_argument("--metric", choices=("ncc", "l2", "edge-ncc"), default="edge-ncc")
    parser.add_argument("--search-radius", type=int, default=15)
    parser.add_argument("--refine-radius", type=int, default=3)
    parser.add_argument("--min-size", type=int, default=400)
    parser.add_argument("--score-crop", type=float, default=0.10)
    parser.add_argument("--border-crop", type=float, default=0.05)
    parser.add_argument("--autocontrast", action="store_true")
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.output is not None and len(args.inputs) != 1:
        raise SystemExit("--output can only be used with one input")

    for input_path in args.inputs:
        output_path = args.output or _default_output(input_path, args.output_dir)
        green_shift, red_shift = colorize(
            input_path=input_path,
            output_path=output_path,
            method=args.method,
            metric=args.metric,
            search_radius=args.search_radius,
            refine_radius=args.refine_radius,
            min_size=args.min_size,
            score_crop=args.score_crop,
            border_crop=args.border_crop,
            autocontrast=args.autocontrast,
        )
        print(f"{input_path.name}")
        print(f"  G -> B: (x={green_shift[0]:+d}, y={green_shift[1]:+d})")
        print(f"  R -> B: (x={red_shift[0]:+d}, y={red_shift[1]:+d})")
        print(f"  saved: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
