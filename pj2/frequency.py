"""Frequency-domain image constructions for Project 2 Part 2."""

from __future__ import annotations

import cv2
import numpy as np
from scipy.signal import convolve2d


def gaussian_kernel(size: int, sigma: float) -> np.ndarray:
    """Return a normalized 2D Gaussian made from OpenCV's 1D kernel."""
    gaussian_1d = cv2.getGaussianKernel(size, sigma)
    return gaussian_1d @ gaussian_1d.T


def convolve_color(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """Convolve every color channel with symmetric boundary extension."""
    if image.ndim == 2:
        return convolve2d(image, kernel, mode="same", boundary="symm")
    return np.stack(
        [
            convolve2d(image[..., channel], kernel, mode="same", boundary="symm")
            for channel in range(image.shape[2])
        ],
        axis=-1,
    )


def unsharp_mask(
    image: np.ndarray,
    kernel_size: int,
    sigma: float,
    amount: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Sharpen with one convolution and return its explanatory components."""
    gaussian = gaussian_kernel(kernel_size, sigma)
    impulse = np.zeros_like(gaussian)
    impulse[kernel_size // 2, kernel_size // 2] = 1.0
    unsharp_filter = (1.0 + amount) * impulse - amount * gaussian

    blurred = convolve_color(image, gaussian)
    high_frequencies = image - blurred
    sharpened = convolve_color(image, unsharp_filter)
    return blurred, high_frequencies, np.clip(sharpened, 0.0, 1.0), unsharp_filter


def align_from_eyes(
    image: np.ndarray,
    left_eye: tuple[float, float],
    right_eye: tuple[float, float],
    output_size: int = 512,
    eye_y: float = 0.38,
    eye_distance: float = 0.26,
) -> np.ndarray:
    """Map two eye landmarks to a shared scale, rotation, and position."""
    source_left = np.asarray(left_eye, dtype=np.float32)
    source_right = np.asarray(right_eye, dtype=np.float32)
    source_midpoint = (source_left + source_right) / 2.0
    source_vector = source_right - source_left
    source_third = source_midpoint + np.array(
        [-source_vector[1], source_vector[0]], dtype=np.float32
    )

    destination_left = np.array(
        [output_size * (0.5 - eye_distance / 2.0), output_size * eye_y],
        dtype=np.float32,
    )
    destination_right = np.array(
        [output_size * (0.5 + eye_distance / 2.0), output_size * eye_y],
        dtype=np.float32,
    )
    destination_midpoint = (destination_left + destination_right) / 2.0
    destination_vector = destination_right - destination_left
    destination_third = destination_midpoint + np.array(
        [-destination_vector[1], destination_vector[0]], dtype=np.float32
    )

    transform = cv2.getAffineTransform(
        np.float32([source_left, source_right, source_third]),
        np.float32([destination_left, destination_right, destination_third]),
    )
    return cv2.warpAffine(
        image,
        transform,
        (output_size, output_size),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REFLECT_101,
    )


def align_to_eye_targets(
    image: np.ndarray,
    left_eye: tuple[float, float],
    right_eye: tuple[float, float],
    target_left_eye: tuple[float, float],
    target_right_eye: tuple[float, float],
    output_width: int,
    output_height: int,
) -> np.ndarray:
    """Similarity-warp an image to explicit eye locations and canvas shape."""
    source_left = np.asarray(left_eye, dtype=np.float32)
    source_right = np.asarray(right_eye, dtype=np.float32)
    source_midpoint = (source_left + source_right) / 2.0
    source_vector = source_right - source_left
    source_third = source_midpoint + np.array(
        [-source_vector[1], source_vector[0]], dtype=np.float32
    )

    target_left = np.asarray(target_left_eye, dtype=np.float32)
    target_right = np.asarray(target_right_eye, dtype=np.float32)
    target_midpoint = (target_left + target_right) / 2.0
    target_vector = target_right - target_left
    target_third = target_midpoint + np.array(
        [-target_vector[1], target_vector[0]], dtype=np.float32
    )

    transform = cv2.getAffineTransform(
        np.float32([source_left, source_right, source_third]),
        np.float32([target_left, target_right, target_third]),
    )
    return cv2.warpAffine(
        image,
        transform,
        (output_width, output_height),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )


def rgb_to_gray(image: np.ndarray) -> np.ndarray:
    return (
        0.2126 * image[..., 0]
        + 0.7152 * image[..., 1]
        + 0.0722 * image[..., 2]
    )


def hybrid_components(
    low_image: np.ndarray,
    high_image: np.ndarray,
    low_size: int,
    low_sigma: float,
    high_size: int,
    high_sigma: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Keep low frequencies from one image and high frequencies from another."""
    low = convolve_color(low_image, gaussian_kernel(low_size, low_sigma))
    high_blur = convolve_color(high_image, gaussian_kernel(high_size, high_sigma))
    high = high_image - high_blur
    return low, high


def compose_hybrid(
    low: np.ndarray,
    high: np.ndarray,
    low_color: bool = True,
    high_color: bool = True,
) -> np.ndarray:
    """Compose a hybrid while independently retaining color in each band."""
    low_component = low if low_color else np.repeat(rgb_to_gray(low)[..., None], 3, axis=2)
    high_component = high if high_color else np.repeat(rgb_to_gray(high)[..., None], 3, axis=2)
    return np.clip(low_component + high_component, 0.0, 1.0)


def log_fourier_magnitude(image: np.ndarray) -> np.ndarray:
    """Return log(1 + |FFT|), centered at zero frequency."""
    gray = rgb_to_gray(image) if image.ndim == 3 else image
    spectrum = np.log1p(np.abs(np.fft.fftshift(np.fft.fft2(gray))))
    low, high = np.percentile(spectrum, (2.0, 99.8))
    return np.clip((spectrum - low) / max(high - low, 1e-12), 0.0, 1.0)


def windowed_fourier_magnitude(image: np.ndarray) -> np.ndarray:
    """Return an FFT magnitude with mean removal and a 2D Hann window."""
    gray = rgb_to_gray(image) if image.ndim == 3 else image
    height, width = gray.shape
    window = np.outer(np.hanning(height), np.hanning(width))
    centered = (gray - np.mean(gray)) * window
    return np.abs(np.fft.fftshift(np.fft.fft2(centered)))


def high_frequency_energy(image: np.ndarray, cutoff: float = 0.10) -> float:
    """Measure spectral energy above a radial cutoff in cycles per pixel."""
    magnitude = windowed_fourier_magnitude(image)
    height, width = magnitude.shape
    vertical_frequency = np.fft.fftshift(np.fft.fftfreq(height))[:, None]
    horizontal_frequency = np.fft.fftshift(np.fft.fftfreq(width))[None, :]
    radius = np.hypot(horizontal_frequency, vertical_frequency)
    return float(np.sum(magnitude[radius >= cutoff] ** 2))
