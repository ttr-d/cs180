"""Generate all reproducible figures and measurements for PJ2 Part 1."""

from __future__ import annotations

import json
import time
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageOps
from scipy.signal import convolve2d

from filter import (
    convolve_four_loops,
    convolve_two_loops,
    gradient_magnitude,
    hsv_to_rgb,
    manual_atan2,
    orientation_rgb,
)


ROOT = Path(__file__).resolve().parent
ASSETS = ROOT / "web_assets"
DX = np.array([[1.0, 0.0, -1.0]])
DY = DX.T
BOX = np.ones((9, 9), dtype=np.float64) / 81.0
GAUSSIAN_SIZE = 9
GAUSSIAN_SIGMA = 1.5
FINITE_THRESHOLD = 0.20
FINITE_THRESHOLD_PREVIEWS = (0.10, 0.15, 0.20, 0.30)
SMOOTH_THRESHOLD = 0.10


def load_grayscale(path: Path, max_width: int | None = None) -> np.ndarray:
    with Image.open(path) as source:
        image = ImageOps.exif_transpose(source).convert("L")
        if max_width and image.width > max_width:
            height = round(image.height * max_width / image.width)
            image = image.resize((max_width, height), Image.Resampling.LANCZOS)
        return np.asarray(image, dtype=np.float64) / 255.0


def save_grayscale(
    array: np.ndarray,
    path: Path,
    *,
    signed: bool = False,
    quality: int = 88,
) -> None:
    values = np.asarray(array, dtype=np.float64)
    if signed:
        limit = max(float(np.percentile(np.abs(values), 99.0)), 1e-12)
        values = 0.5 + 0.5 * np.clip(values / limit, -1.0, 1.0)
    else:
        values = np.clip(values, 0.0, 1.0)
    image = Image.fromarray(np.round(values * 255).astype(np.uint8))
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() in {".jpg", ".jpeg"}:
        image.save(path, quality=quality, optimize=True, progressive=True)
    else:
        image.save(path, optimize=True)


def save_rgb(array: np.ndarray, path: Path, quality: int = 88) -> None:
    values = np.clip(np.asarray(array), 0.0, 1.0)
    image = Image.fromarray(np.round(values * 255).astype(np.uint8))
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path, quality=quality, optimize=True, progressive=True)


def optimize_source_photo(path: Path, max_width: int = 1600) -> None:
    """Safely replace an oversized source photo with a sub-1 MB copy."""
    if path.stat().st_size < 1_000_000:
        return
    temporary = path.with_name(f".{path.stem}-optimized{path.suffix}")
    with Image.open(path) as source:
        image = ImageOps.exif_transpose(source).convert("RGB")
        if image.width > max_width:
            height = round(image.height * max_width / image.width)
            image = image.resize((max_width, height), Image.Resampling.LANCZOS)
        quality = 90
        while quality >= 60:
            image.save(temporary, quality=quality, optimize=True, progressive=True)
            if temporary.stat().st_size < 1_000_000:
                break
            quality -= 4
    if temporary.stat().st_size >= 1_000_000:
        temporary.unlink()
        raise RuntimeError(f"Could not optimize {path.name} below 1 MB")
    temporary.replace(path)


def save_kernel_figure(kernel: np.ndarray, path: Path, title: str, cmap: str) -> None:
    figure, axis = plt.subplots(figsize=(4.2, 4.0), dpi=160)
    limit = float(np.max(np.abs(kernel)))
    kwargs = {"cmap": cmap}
    if np.min(kernel) < 0:
        kwargs.update(vmin=-limit, vmax=limit)
    image = axis.imshow(kernel, **kwargs)
    axis.set_title(title, fontsize=12, pad=10)
    axis.set_xticks(range(kernel.shape[1]))
    axis.set_yticks(range(kernel.shape[0]))
    axis.tick_params(labelsize=7)
    figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, bbox_inches="tight", facecolor="#f4f0e8")
    plt.close(figure)


def benchmark() -> dict[str, float | int]:
    rng = np.random.default_rng(180)
    sample = rng.random((128, 128))

    # Warm-up calls avoid measuring one-time import/allocation effects.
    convolve_two_loops(sample[:8, :8], BOX)
    convolve2d(sample[:8, :8], BOX, mode="same", boundary="fill", fillvalue=0)

    def measure(function, repeats: int) -> float:
        timings = []
        for _ in range(repeats):
            start = time.perf_counter()
            function()
            timings.append((time.perf_counter() - start) * 1000.0)
        return float(np.median(timings))

    four_ms = measure(lambda: convolve_four_loops(sample, BOX), 3)
    two_ms = measure(lambda: convolve_two_loops(sample, BOX), 5)
    scipy_ms = measure(
        lambda: convolve2d(sample, BOX, mode="same", boundary="fill", fillvalue=0),
        15,
    )
    return {
        "benchmark_size": 128,
        "four_loop_ms": round(four_ms, 2),
        "two_loop_ms": round(two_ms, 2),
        "scipy_ms": round(scipy_ms, 2),
    }


def verify_convolution() -> dict[str, float]:
    rng = np.random.default_rng(42)
    image = rng.normal(size=(13, 17))
    kernel = rng.normal(size=(4, 5))
    metrics: dict[str, float] = {}
    for mode in ("same", "full"):
        expected = convolve2d(image, kernel, mode=mode, boundary="fill", fillvalue=0)
        four = convolve_four_loops(image, kernel, mode=mode)
        two = convolve_two_loops(image, kernel, mode=mode)
        metrics[f"four_loop_{mode}_error"] = float(np.max(np.abs(four - expected)))
        metrics[f"two_loop_{mode}_error"] = float(np.max(np.abs(two - expected)))
    return metrics


def main() -> None:
    convolution_dir = ASSETS / "convolution"
    finite_dir = ASSETS / "finite-difference"
    gaussian_dir = ASSETS / "gaussian"
    dog_dir = ASSETS / "dog"
    orientation_dir = ASSETS / "orientation"
    for directory in (
        convolution_dir,
        finite_dir,
        gaussian_dir,
        dog_dir,
        orientation_dir,
    ):
        directory.mkdir(parents=True, exist_ok=True)

    # Part 1.1: run the NumPy-only two-loop implementation on a manageable,
    # high-quality working copy.  The source is optimized once if it exceeds 1 MB.
    optimize_source_photo(ROOT / "selfie.jpg")
    selfie = load_grayscale(ROOT / "selfie.jpg", max_width=720)
    save_grayscale(selfie, convolution_dir / "selfie-grayscale.jpg")
    selfie_box = convolve_two_loops(selfie, BOX)
    selfie_dx = convolve_two_loops(selfie, DX)
    selfie_dy = convolve_two_loops(selfie, DY)
    save_grayscale(selfie_box, convolution_dir / "selfie-box-9x9.jpg")
    save_grayscale(selfie_dx, convolution_dir / "selfie-dx.jpg", signed=True)
    save_grayscale(selfie_dy, convolution_dir / "selfie-dy.jpg", signed=True)

    # Part 1.2: finite differences on cameraman.
    cameraman = load_grayscale(ROOT / "cameraman.png")
    save_grayscale(cameraman, finite_dir / "cameraman.jpg")
    finite_dx = convolve2d(cameraman, DX, mode="same", boundary="fill", fillvalue=0)
    finite_dy = convolve2d(cameraman, DY, mode="same", boundary="fill", fillvalue=0)
    finite_magnitude = gradient_magnitude(finite_dx, finite_dy)
    finite_edges = finite_magnitude >= FINITE_THRESHOLD
    save_grayscale(finite_dx, finite_dir / "dx.jpg", signed=True)
    save_grayscale(finite_dy, finite_dir / "dy.jpg", signed=True)
    save_grayscale(finite_magnitude / max(float(finite_magnitude.max()), 1e-12), finite_dir / "magnitude.jpg")
    save_grayscale(finite_edges.astype(float), finite_dir / "edges.jpg")
    for threshold in FINITE_THRESHOLD_PREVIEWS:
        threshold_code = round(threshold * 100)
        threshold_edges = finite_magnitude >= threshold
        save_grayscale(
            threshold_edges.astype(float),
            finite_dir / f"edges-{threshold_code:03d}.jpg",
        )

    # Part 1.3: Gaussian smoothing followed by finite differences.
    gaussian_1d = cv2.getGaussianKernel(GAUSSIAN_SIZE, GAUSSIAN_SIGMA)
    gaussian = gaussian_1d @ gaussian_1d.T
    save_kernel_figure(gaussian, gaussian_dir / "gaussian-kernel.png", "9 × 9 Gaussian, σ = 1.5", "viridis")
    blurred = convolve2d(cameraman, gaussian, mode="same", boundary="fill", fillvalue=0)
    smooth_dx = convolve2d(blurred, DX, mode="same", boundary="fill", fillvalue=0)
    smooth_dy = convolve2d(blurred, DY, mode="same", boundary="fill", fillvalue=0)
    smooth_magnitude = gradient_magnitude(smooth_dx, smooth_dy)
    smooth_edges = smooth_magnitude >= SMOOTH_THRESHOLD
    save_grayscale(blurred, gaussian_dir / "blurred.jpg")
    save_grayscale(smooth_dx, gaussian_dir / "smooth-dx.jpg", signed=True)
    save_grayscale(smooth_dy, gaussian_dir / "smooth-dy.jpg", signed=True)
    save_grayscale(smooth_magnitude / max(float(smooth_magnitude.max()), 1e-12), gaussian_dir / "smooth-magnitude.jpg")
    save_grayscale(smooth_edges.astype(float), gaussian_dir / "smooth-edges.jpg")

    # Part 1.4: convolve once with derivative-of-Gaussian filters.
    dog_x = convolve2d(gaussian, DX, mode="full", boundary="fill", fillvalue=0)
    dog_y = convolve2d(gaussian, DY, mode="full", boundary="fill", fillvalue=0)
    save_kernel_figure(dog_x, dog_dir / "dog-x-filter.png", "Derivative of Gaussian — x", "coolwarm")
    save_kernel_figure(dog_y, dog_dir / "dog-y-filter.png", "Derivative of Gaussian — y", "coolwarm")
    dog_dx = convolve2d(cameraman, dog_x, mode="same", boundary="fill", fillvalue=0)
    dog_dy = convolve2d(cameraman, dog_y, mode="same", boundary="fill", fillvalue=0)
    dog_magnitude = gradient_magnitude(dog_dx, dog_dy)
    dog_edges = dog_magnitude >= SMOOTH_THRESHOLD
    save_grayscale(dog_dx, dog_dir / "dog-dx.jpg", signed=True)
    save_grayscale(dog_dy, dog_dir / "dog-dy.jpg", signed=True)
    save_grayscale(dog_magnitude / max(float(dog_magnitude.max()), 1e-12), dog_dir / "dog-magnitude.jpg")
    save_grayscale(dog_edges.astype(float), dog_dir / "dog-edges.jpg")
    difference = np.abs(dog_magnitude - smooth_magnitude)
    difference_scale = max(float(np.percentile(difference, 99.5)), 1e-12)
    save_grayscale(np.clip(difference / difference_scale, 0, 1), dog_dir / "path-difference.png")

    # Part 1.4: no built-in angle function is used to create these results.
    # Save both the continuous magnitude view and the edge-masked comparison.
    orientation_all = orientation_rgb(dog_dx, dog_dy)
    orientation_edges = orientation_rgb(
        dog_dx, dog_dy, threshold=SMOOTH_THRESHOLD
    )
    save_rgb(orientation_all, orientation_dir / "orientation-all.jpg", quality=92)
    save_rgb(orientation_edges, orientation_dir / "orientation.jpg", quality=92)
    hue_strip = np.linspace(0.0, 1.0, 720, endpoint=False)[None, :]
    wheel = np.repeat(
        hsv_to_rgb(hue_strip, np.ones_like(hue_strip), np.ones_like(hue_strip)),
        54,
        axis=0,
    )
    save_rgb(wheel, orientation_dir / "orientation-key.jpg", quality=94)

    border = GAUSSIAN_SIZE // 2 + 1
    interior = np.s_[border:-border, border:-border]
    orientation_reference = np.arctan2(dog_dy, dog_dx)
    wrapped_error = np.abs(
        ((manual_atan2(dog_dy, dog_dx) - orientation_reference + np.pi) % (2 * np.pi))
        - np.pi
    )
    orientation_edge_mask = dog_magnitude >= SMOOTH_THRESHOLD
    interior_orientation_error = wrapped_error[interior][orientation_edge_mask[interior]]
    metrics: dict[str, float | int] = {
        **verify_convolution(),
        **benchmark(),
        "selfie_source_bytes": (ROOT / "selfie.jpg").stat().st_size,
        "finite_threshold": FINITE_THRESHOLD,
        "smooth_threshold": SMOOTH_THRESHOLD,
        "dog_interior_mae": float(np.mean(np.abs(dog_magnitude[interior] - smooth_magnitude[interior]))),
        "dog_interior_max_error": float(np.max(np.abs(dog_magnitude[interior] - smooth_magnitude[interior]))),
        "orientation_mean_error_degrees": float(np.degrees(np.mean(interior_orientation_error))),
        "orientation_max_error_degrees": float(np.degrees(np.max(interior_orientation_error))),
    }
    with (ASSETS / "metrics.json").open("w", encoding="utf-8") as output:
        json.dump(metrics, output, indent=2)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
