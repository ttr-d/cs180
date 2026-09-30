"""Gaussian and Laplacian stacks without any spatial downsampling."""

from __future__ import annotations

import math

import cv2
import numpy as np
from scipy.ndimage import convolve1d


def gaussian_blur(image: np.ndarray, sigma: float) -> np.ndarray:
    """Apply a separable Gaussian convolution with reflected boundaries."""
    kernel_size = 2 * math.ceil(3 * sigma) + 1
    kernel_1d = cv2.getGaussianKernel(kernel_size, sigma).ravel()
    vertical = convolve1d(image, kernel_1d, axis=0, mode="reflect")
    return convolve1d(vertical, kernel_1d, axis=1, mode="reflect")


def srgb_to_linear(image: np.ndarray) -> np.ndarray:
    """Decode sRGB values so filtering and addition happen in linear light."""
    values = np.clip(np.asarray(image, dtype=np.float64), 0.0, 1.0)
    return np.where(
        values <= 0.04045,
        values / 12.92,
        ((values + 0.055) / 1.055) ** 2.4,
    )


def linear_to_srgb(image: np.ndarray) -> np.ndarray:
    """Encode linear-light RGB values for display."""
    values = np.clip(np.asarray(image, dtype=np.float64), 0.0, 1.0)
    return np.where(
        values <= 0.0031308,
        12.92 * values,
        1.055 * (values ** (1.0 / 2.4)) - 0.055,
    )


def gaussian_stack(
    image: np.ndarray,
    levels: int = 5,
    base_sigma: float = 1.0,
) -> list[np.ndarray]:
    """Build equally sized Gaussian levels with progressively wider blur."""
    if levels < 1:
        raise ValueError("levels must be at least one")

    stack = [np.asarray(image, dtype=np.float64)]
    for level in range(1, levels):
        sigma = base_sigma * (2 ** (level - 1))
        stack.append(gaussian_blur(stack[-1], sigma))
    return stack


def laplacian_stack(gaussian_levels: list[np.ndarray]) -> list[np.ndarray]:
    """Store adjacent Gaussian differences plus the final low-pass residual."""
    if not gaussian_levels:
        raise ValueError("gaussian_levels cannot be empty")
    differences = [
        gaussian_levels[level] - gaussian_levels[level + 1]
        for level in range(len(gaussian_levels) - 1)
    ]
    return differences + [gaussian_levels[-1]]


def blend_laplacian_stacks(
    left_levels: list[np.ndarray],
    right_levels: list[np.ndarray],
    mask_gaussian_levels: list[np.ndarray],
) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray]]:
    """Blend corresponding Laplacian levels with a blurred mask at each scale."""
    if not (
        len(left_levels) == len(right_levels) == len(mask_gaussian_levels)
    ):
        raise ValueError("image and mask stacks must have equal length")

    masked_left = []
    masked_right = []
    blended = []
    for left, right, mask in zip(
        left_levels, right_levels, mask_gaussian_levels
    ):
        expanded_mask = mask[..., None] if mask.ndim == 2 else mask
        left_part = expanded_mask * left
        right_part = (1.0 - expanded_mask) * right
        masked_left.append(left_part)
        masked_right.append(right_part)
        blended.append(left_part + right_part)
    return masked_left, masked_right, blended
