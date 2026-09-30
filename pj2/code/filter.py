"""Core filtering routines for CS 180 Project 2, Part 1.

The two convolution implementations intentionally use only NumPy.  SciPy is
used by the asset-generation script solely as a reference implementation.
"""

from __future__ import annotations

from typing import Literal

import numpy as np


Mode = Literal["same", "full"]


def _prepare_convolution(
    image: np.ndarray, kernel: np.ndarray, mode: Mode
) -> tuple[np.ndarray, np.ndarray, tuple[int, int]]:
    """Return a padded image, flipped kernel, and output shape."""
    image = np.asarray(image, dtype=np.float64)
    kernel = np.asarray(kernel, dtype=np.float64)
    if image.ndim != 2 or kernel.ndim != 2:
        raise ValueError("image and kernel must both be two-dimensional")
    if image.size == 0 or kernel.size == 0:
        raise ValueError("image and kernel must be non-empty")
    if mode not in {"same", "full"}:
        raise ValueError("mode must be 'same' or 'full'")

    height, width = image.shape
    kernel_height, kernel_width = kernel.shape
    if mode == "full":
        pad_top = pad_bottom = kernel_height - 1
        pad_left = pad_right = kernel_width - 1
        output_shape = (height + kernel_height - 1, width + kernel_width - 1)
    else:
        # This asymmetric split for even kernels matches convolve2d(mode="same").
        pad_top = kernel_height // 2
        pad_bottom = kernel_height - 1 - pad_top
        pad_left = kernel_width // 2
        pad_right = kernel_width - 1 - pad_left
        output_shape = image.shape

    padded = np.pad(
        image,
        ((pad_top, pad_bottom), (pad_left, pad_right)),
        mode="constant",
        constant_values=0,
    )
    flipped = kernel[::-1, ::-1]
    return padded, flipped, output_shape


def convolve_four_loops(
    image: np.ndarray, kernel: np.ndarray, mode: Mode = "same"
) -> np.ndarray:
    """Convolve a 2-D image with four explicit Python loops."""
    padded, flipped, (output_height, output_width) = _prepare_convolution(
        image, kernel, mode
    )
    kernel_height, kernel_width = flipped.shape
    output = np.zeros((output_height, output_width), dtype=np.float64)

    for row in range(output_height):
        for column in range(output_width):
            value = 0.0
            for kernel_row in range(kernel_height):
                for kernel_column in range(kernel_width):
                    value += (
                        padded[row + kernel_row, column + kernel_column]
                        * flipped[kernel_row, kernel_column]
                    )
            output[row, column] = value
    return output


def convolve_two_loops(
    image: np.ndarray, kernel: np.ndarray, mode: Mode = "same"
) -> np.ndarray:
    """Convolve a 2-D image using two loops and a vectorized inner product."""
    padded, flipped, (output_height, output_width) = _prepare_convolution(
        image, kernel, mode
    )
    kernel_height, kernel_width = flipped.shape
    output = np.empty((output_height, output_width), dtype=np.float64)

    for row in range(output_height):
        for column in range(output_width):
            window = padded[
                row : row + kernel_height,
                column : column + kernel_width,
            ]
            output[row, column] = np.sum(window * flipped)
    return output


def gradient_components(
    image: np.ndarray,
    convolution=convolve_two_loops,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute horizontal and vertical central finite differences."""
    dx_filter = np.array([[1.0, 0.0, -1.0]])
    dy_filter = dx_filter.T
    return (
        convolution(image, dx_filter, mode="same"),
        convolution(image, dy_filter, mode="same"),
    )


def gradient_magnitude(dx: np.ndarray, dy: np.ndarray) -> np.ndarray:
    """Return sqrt(dx^2 + dy^2) without changing the input arrays."""
    return np.sqrt(np.square(dx) + np.square(dy))


def manual_atan2(y: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Approximate atan2(y, x) using ratios, a polynomial, and quadrants.

    This function deliberately avoids np.arctan, np.arctan2, and every other
    built-in angle function.  The returned range is [-pi, pi].
    """
    y, x = np.broadcast_arrays(
        np.asarray(y, dtype=np.float64), np.asarray(x, dtype=np.float64)
    )
    abs_x = np.abs(x)
    abs_y = np.abs(y)
    y_is_dominant = abs_y > abs_x
    larger = np.maximum(abs_x, abs_y)
    ratio = np.divide(
        np.minimum(abs_x, abs_y),
        larger,
        out=np.zeros_like(larger),
        where=larger > 0,
    )
    # Compact atan approximation on [0, 1]; max error is about 0.0015 rad.
    base = (
        (np.pi / 4.0) * ratio
        - ratio * (ratio - 1.0) * (0.2447 + 0.0663 * ratio)
    )
    first_quadrant = np.where(
        y_is_dominant, np.pi / 2.0 - base, base
    )

    angle = np.where(
        x >= 0,
        np.where(y >= 0, first_quadrant, -first_quadrant),
        np.where(y >= 0, np.pi - first_quadrant, first_quadrant - np.pi),
    )
    return np.where((x == 0) & (y == 0), 0.0, angle)


def hsv_to_rgb(hue: np.ndarray, saturation: np.ndarray, value: np.ndarray) -> np.ndarray:
    """Vectorized HSV-to-RGB conversion for values in [0, 1]."""
    hue, saturation, value = np.broadcast_arrays(hue, saturation, value)
    sector_position = (hue % 1.0) * 6.0
    sector = np.floor(sector_position).astype(np.int8)
    fraction = sector_position - sector
    p = value * (1.0 - saturation)
    q = value * (1.0 - saturation * fraction)
    t = value * (1.0 - saturation * (1.0 - fraction))

    rgb_choices = (
        (value, t, p),
        (q, value, p),
        (p, value, t),
        (p, q, value),
        (t, p, value),
        (value, p, q),
    )
    output = np.empty(hue.shape + (3,), dtype=np.float64)
    for index, channels in enumerate(rgb_choices):
        mask = sector == index
        output[mask] = np.stack(channels, axis=-1)[mask]
    return output


def orientation_rgb(
    dx: np.ndarray, dy: np.ndarray, threshold: float = 0.0
) -> np.ndarray:
    """Encode direction as hue, while showing only sufficiently strong edges."""
    magnitude = gradient_magnitude(dx, dy)
    scale = max(float(np.percentile(magnitude, 99.0)), np.finfo(float).eps)
    value = np.clip(magnitude / scale, 0.0, 1.0) ** 0.55
    value = np.where(magnitude >= threshold, value, 0.0)
    hue = (manual_atan2(dy, dx) + np.pi) / (2.0 * np.pi)
    return hsv_to_rgb(hue, np.ones_like(hue), value)
