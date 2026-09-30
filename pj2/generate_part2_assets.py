"""Generate reproducible figures for PJ2 Parts 2.1 and 2.2."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps

from frequency import (
    align_from_eyes,
    align_to_eye_targets,
    compose_hybrid,
    convolve_color,
    gaussian_kernel,
    high_frequency_energy,
    hybrid_components,
    log_fourier_magnitude,
    rgb_to_gray,
    unsharp_mask,
    windowed_fourier_magnitude,
)
from stacks import (
    blend_laplacian_stacks,
    gaussian_blur,
    gaussian_stack,
    laplacian_stack,
    linear_to_srgb,
    srgb_to_linear,
)


ROOT = Path(__file__).resolve().parent
ASSETS = ROOT / "web_assets"
SOURCE_IMAGES = ROOT / "source_images"


def load_rgb(path: Path) -> np.ndarray:
    with Image.open(path) as source:
        return np.asarray(ImageOps.exif_transpose(source).convert("RGB"), dtype=np.float64) / 255.0


def load_binary_mask(path: Path) -> np.ndarray:
    with Image.open(path) as source:
        grayscale = ImageOps.exif_transpose(source).convert("L")
        return (np.asarray(grayscale, dtype=np.float64) >= 127.5).astype(np.float64)


def save_rgb(array: np.ndarray, path: Path, quality: int = 90) -> None:
    values = np.clip(np.asarray(array), 0.0, 1.0)
    image = Image.fromarray(np.round(values * 255.0).astype(np.uint8))
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path, quality=quality, optimize=True, progressive=True)


def save_gray(array: np.ndarray, path: Path, quality: int = 90) -> None:
    values = np.clip(np.asarray(array), 0.0, 1.0)
    image = Image.fromarray(np.round(values * 255.0).astype(np.uint8))
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path, quality=quality, optimize=True, progressive=True)


def resize_rgb(image: np.ndarray, width: int, height: int) -> np.ndarray:
    """Resize an RGB float image, using area resampling when reducing it."""
    interpolation = (
        cv2.INTER_AREA
        if height <= image.shape[0] and width <= image.shape[1]
        else cv2.INTER_CUBIC
    )
    return cv2.resize(image, (width, height), interpolation=interpolation)


def resize_long_edge(image: np.ndarray, long_edge: int) -> np.ndarray:
    """Resize an RGB image while retaining its original aspect ratio."""
    height, width = image.shape[:2]
    scale = long_edge / max(height, width)
    return resize_rgb(image, round(width * scale), round(height * scale))


def save_high_frequency(array: np.ndarray, path: Path) -> None:
    # Zero is middle gray, so both positive and negative detail remain visible.
    scale = max(float(np.percentile(np.abs(array), 99.5)), 1e-12)
    save_rgb(0.5 + 0.5 * np.clip(array / scale, -1.0, 1.0), path)


def save_stack_level(
    array: np.ndarray,
    path: Path,
    *,
    signed_scale: float | None = None,
) -> None:
    if signed_scale is None:
        save_rgb(array, path, quality=92)
    else:
        display = 0.5 + 0.5 * np.clip(array / max(signed_scale, 1e-12), -1.0, 1.0)
        save_rgb(display, path, quality=92)


def save_shared_spectra(images: list[np.ndarray], paths: list[Path]) -> None:
    """Save FFT plots using one display scale so brightness is comparable."""
    log_spectra = [np.log1p(windowed_fourier_magnitude(image)) for image in images]
    floor = min(float(np.percentile(spectrum, 2.0)) for spectrum in log_spectra)
    ceiling = max(float(np.percentile(spectrum, 99.8)) for spectrum in log_spectra)
    scale = max(ceiling - floor, 1e-12)
    for spectrum, path in zip(log_spectra, paths):
        save_gray(np.clip((spectrum - floor) / scale, 0.0, 1.0), path)


def generate_sharpening() -> dict[str, float]:
    output = ASSETS / "sharpening"
    output.mkdir(parents=True, exist_ok=True)

    taj = load_rgb(SOURCE_IMAGES / "sharpening" / "taj.jpg")
    taj_blur, taj_high, taj_sharp, taj_filter = unsharp_mask(
        taj, kernel_size=13, sigma=2.0, amount=1.5
    )
    _, _, taj_strong, _ = unsharp_mask(
        taj, kernel_size=13, sigma=2.0, amount=3.0
    )
    save_rgb(taj, output / "taj-original.jpg")
    save_rgb(taj_blur, output / "taj-low-pass.jpg")
    save_high_frequency(taj_high, output / "taj-high-pass.jpg")
    save_rgb(taj_sharp, output / "taj-sharpened.jpg", quality=94)
    save_rgb(taj_strong, output / "taj-sharpened-strong.jpg", quality=94)

    # Verify that the single unsharp filter matches the algebraic two-step form.
    taj_two_step = np.clip(taj + 1.5 * (taj - taj_blur), 0.0, 1.0)
    single_step_error = float(np.max(np.abs(taj_sharp - taj_two_step)))

    # Evaluation starts from a naturally sharp image, removes frequencies, then
    # attempts to restore contrast around the surviving transitions.
    evaluation_original = align_from_eyes(
        load_rgb(SOURCE_IMAGES / "hybrids" / "raccoon.jpg"),
        left_eye=(322, 281),
        right_eye=(433, 281),
        output_size=512,
        eye_y=0.37,
        eye_distance=0.25,
    )
    evaluation_blur = convolve_color(evaluation_original, gaussian_kernel(17, 2.8))
    _, _, evaluation_recovered, _ = unsharp_mask(
        evaluation_blur, kernel_size=17, sigma=2.8, amount=2.0
    )
    save_rgb(evaluation_original, output / "evaluation-original.jpg", quality=92)
    save_rgb(evaluation_blur, output / "evaluation-blurred.jpg", quality=92)
    save_rgb(evaluation_recovered, output / "evaluation-resharpened.jpg", quality=92)

    save_shared_spectra(
        [evaluation_original, evaluation_blur, evaluation_recovered],
        [
            output / "evaluation-fft-original.jpg",
            output / "evaluation-fft-blurred.jpg",
            output / "evaluation-fft-resharpened.jpg",
        ],
    )
    original_high_energy = high_frequency_energy(evaluation_original)
    blur_high_energy = high_frequency_energy(evaluation_blur)
    recovered_high_energy = high_frequency_energy(evaluation_recovered)

    return {
        "unsharp_single_step_max_error": single_step_error,
        "unsharp_kernel_sum": float(np.sum(taj_filter)),
        "evaluation_blur_mae": float(np.mean(np.abs(evaluation_blur - evaluation_original))),
        "evaluation_recovered_mae": float(np.mean(np.abs(evaluation_recovered - evaluation_original))),
        "evaluation_original_hf_percent": 100.0,
        "evaluation_blur_hf_percent": 100.0 * blur_high_energy / original_high_energy,
        "evaluation_recovered_hf_percent": 100.0 * recovered_high_energy / original_high_energy,
    }


def save_frequency_set(
    directory: Path,
    high_input: np.ndarray,
    low_input: np.ndarray,
    high_component: np.ndarray,
    low_component: np.ndarray,
    hybrid: np.ndarray,
) -> None:
    save_gray(log_fourier_magnitude(high_input), directory / "fft-high-input.jpg")
    save_gray(log_fourier_magnitude(low_input), directory / "fft-low-input.jpg")
    save_gray(log_fourier_magnitude(high_component), directory / "fft-high-filtered.jpg")
    save_gray(log_fourier_magnitude(low_component), directory / "fft-low-filtered.jpg")
    save_gray(log_fourier_magnitude(hybrid), directory / "fft-hybrid.jpg")


def generate_hybrids() -> dict[str, float]:
    derek_dir = ASSETS / "hybrid-derek-nutmeg"
    einstein_dir = ASSETS / "hybrid-einstein-marilyn"
    animal_dir = ASSETS / "hybrid-raccoon-wolf"
    derek_dir.mkdir(parents=True, exist_ok=True)
    einstein_dir.mkdir(parents=True, exist_ok=True)
    animal_dir.mkdir(parents=True, exist_ok=True)

    # Keep Derek upright and preserve his original 732:1024 portrait ratio.
    output_height = 512
    output_width = round(output_height * 732 / 1024)
    derek_source = load_rgb(SOURCE_IMAGES / "hybrids" / "DerekPicture.jpg")
    derek = cv2.resize(
        derek_source, (output_width, output_height), interpolation=cv2.INTER_AREA
    )
    horizontal_scale = output_width / derek_source.shape[1]
    vertical_scale = output_height / derek_source.shape[0]
    derek_left_eye = (299 * horizontal_scale, 343 * vertical_scale)
    derek_right_eye = (439 * horizontal_scale, 330 * vertical_scale)
    nutmeg = align_to_eye_targets(
        load_rgb(SOURCE_IMAGES / "hybrids" / "nutmeg.jpg"),
        left_eye=(607, 285),
        right_eye=(751, 363),
        target_left_eye=derek_left_eye,
        target_right_eye=derek_right_eye,
        output_width=output_width,
        output_height=output_height,
    )
    derek_low, nutmeg_high = hybrid_components(
        derek, nutmeg,
        low_size=41, low_sigma=7.0,
        high_size=25, high_sigma=4.0,
    )
    derek_hybrid = compose_hybrid(
        derek_low, nutmeg_high, low_color=True, high_color=False
    )
    save_rgb(derek, derek_dir / "derek-aligned.jpg", quality=92)
    save_rgb(nutmeg, derek_dir / "nutmeg-aligned.jpg", quality=92)
    save_rgb(derek_low, derek_dir / "derek-low.jpg", quality=92)
    save_high_frequency(nutmeg_high, derek_dir / "nutmeg-high.jpg")
    save_rgb(derek_hybrid, derek_dir / "hybrid.jpg", quality=94)

    einstein = align_from_eyes(
        load_rgb(SOURCE_IMAGES / "hybrids" / "iestein.png"),
        left_eye=(86, 110),
        right_eye=(141, 109),
        output_size=512,
        eye_y=0.38,
        eye_distance=0.25,
    )
    marilyn = align_from_eyes(
        load_rgb(SOURCE_IMAGES / "hybrids" / "marilyn.png"),
        left_eye=(82, 109),
        right_eye=(145, 110),
        output_size=512,
        eye_y=0.38,
        eye_distance=0.25,
    )
    marilyn_low, einstein_high = hybrid_components(
        marilyn, einstein,
        low_size=41, low_sigma=8.0,
        high_size=19, high_sigma=3.0,
    )
    einstein_hybrid_gray = compose_hybrid(
        marilyn_low, einstein_high, low_color=False, high_color=False
    )
    save_rgb(einstein, einstein_dir / "einstein-aligned.jpg")
    save_rgb(marilyn, einstein_dir / "marilyn-aligned.jpg")
    save_rgb(marilyn_low, einstein_dir / "marilyn-low.jpg")
    save_high_frequency(einstein_high, einstein_dir / "einstein-high.jpg")
    save_rgb(einstein_hybrid_gray, einstein_dir / "hybrid-gray.jpg", quality=94)
    save_frequency_set(
        einstein_dir,
        einstein,
        marilyn,
        einstein_high,
        marilyn_low,
        einstein_hybrid_gray,
    )

    raccoon = align_from_eyes(
        load_rgb(SOURCE_IMAGES / "hybrids" / "raccoon.jpg"),
        left_eye=(322, 281),
        right_eye=(433, 281),
        output_size=512,
        eye_y=0.37,
        eye_distance=0.25,
    )
    wolf = align_from_eyes(
        load_rgb(SOURCE_IMAGES / "hybrids" / "wolf.jpg"),
        left_eye=(191, 179),
        right_eye=(276, 179),
        output_size=512,
        eye_y=0.37,
        eye_distance=0.25,
    )
    wolf_low, raccoon_high = hybrid_components(
        wolf, raccoon,
        low_size=45, low_sigma=9.0,
        high_size=25, high_sigma=4.5,
    )
    animal_hybrid_gray = compose_hybrid(
        wolf_low, raccoon_high, low_color=False, high_color=False
    )
    animal_hybrid_low_color = compose_hybrid(
        wolf_low, raccoon_high, low_color=True, high_color=False
    )
    animal_hybrid_high_color = compose_hybrid(
        wolf_low, raccoon_high, low_color=False, high_color=True
    )
    animal_hybrid_both_color = compose_hybrid(
        wolf_low, raccoon_high, low_color=True, high_color=True
    )
    save_rgb(raccoon, animal_dir / "raccoon-aligned.jpg")
    save_rgb(wolf, animal_dir / "wolf-aligned.jpg")
    save_rgb(animal_hybrid_low_color, animal_dir / "hybrid.jpg", quality=94)
    save_rgb(animal_hybrid_gray, animal_dir / "hybrid-gray.jpg", quality=94)
    save_rgb(animal_hybrid_low_color, animal_dir / "hybrid-low-color.jpg", quality=94)
    save_rgb(animal_hybrid_high_color, animal_dir / "hybrid-high-color.jpg", quality=94)
    save_rgb(animal_hybrid_both_color, animal_dir / "hybrid-both-color.jpg", quality=94)

    return {
        "color_high_effect_gray_low_mae": float(
            np.mean(np.abs(animal_hybrid_high_color - animal_hybrid_gray))
        ),
        "color_high_effect_color_low_mae": float(
            np.mean(np.abs(animal_hybrid_both_color - animal_hybrid_low_color))
        ),
        "color_low_effect_gray_high_mae": float(
            np.mean(np.abs(animal_hybrid_low_color - animal_hybrid_gray))
        ),
        "color_low_effect_color_high_mae": float(
            np.mean(np.abs(animal_hybrid_both_color - animal_hybrid_high_color))
        ),
    }


def generate_stacks() -> dict[str, float | int]:
    output = ASSETS / "stacks-oraple"
    output.mkdir(parents=True, exist_ok=True)

    apple = load_rgb(SOURCE_IMAGES / "stacks" / "apple.jpeg")
    orange = load_rgb(SOURCE_IMAGES / "stacks" / "orange.jpeg")
    if apple.shape != orange.shape:
        raise ValueError("apple and orange must have the same dimensions")

    levels = 5
    apple_gaussian = gaussian_stack(apple, levels=levels, base_sigma=1.0)
    orange_gaussian = gaussian_stack(orange, levels=levels, base_sigma=1.0)
    apple_laplacian = laplacian_stack(apple_gaussian)
    orange_laplacian = laplacian_stack(orange_gaussian)

    mask = np.zeros(apple.shape[:2], dtype=np.float64)
    mask[:, : apple.shape[1] // 2] = 1.0
    mask_gaussian = gaussian_stack(mask, levels=levels, base_sigma=1.0)
    masked_apple, masked_orange, blended_levels = blend_laplacian_stacks(
        apple_laplacian, orange_laplacian, mask_gaussian
    )
    apple_contribution = np.clip(np.sum(masked_apple, axis=0), 0.0, 1.0)
    orange_contribution = np.clip(np.sum(masked_orange, axis=0), 0.0, 1.0)
    oraple = np.clip(np.sum(blended_levels, axis=0), 0.0, 1.0)

    for level in range(levels):
        save_rgb(apple_gaussian[level], output / f"apple-gaussian-{level}.jpg")
        save_rgb(orange_gaussian[level], output / f"orange-gaussian-{level}.jpg")

        if level < levels - 1:
            stack_scale = max(
                float(np.percentile(np.abs(apple_laplacian[level]), 99.5)),
                float(np.percentile(np.abs(orange_laplacian[level]), 99.5)),
                1e-12,
            )
            blend_scale = max(
                float(np.percentile(np.abs(masked_apple[level]), 99.5)),
                float(np.percentile(np.abs(masked_orange[level]), 99.5)),
                float(np.percentile(np.abs(blended_levels[level]), 99.5)),
                1e-12,
            )
        else:
            stack_scale = None
            blend_scale = None

        save_stack_level(
            apple_laplacian[level],
            output / f"apple-laplacian-{level}.jpg",
            signed_scale=stack_scale,
        )
        save_stack_level(
            orange_laplacian[level],
            output / f"orange-laplacian-{level}.jpg",
            signed_scale=stack_scale,
        )
        save_stack_level(
            masked_apple[level],
            output / f"blend-apple-{level}.jpg",
            signed_scale=blend_scale,
        )
        save_stack_level(
            masked_orange[level],
            output / f"blend-orange-{level}.jpg",
            signed_scale=blend_scale,
        )
        save_stack_level(
            blended_levels[level],
            output / f"blend-combined-{level}.jpg",
            signed_scale=blend_scale,
        )

    save_rgb(apple_contribution, output / "oraple-apple-contribution.jpg", quality=94)
    save_rgb(orange_contribution, output / "oraple-orange-contribution.jpg", quality=94)
    save_rgb(oraple, output / "oraple.jpg", quality=94)
    apple_reconstruction = np.sum(apple_laplacian, axis=0)
    orange_reconstruction = np.sum(orange_laplacian, axis=0)
    return {
        "stack_levels": levels,
        "apple_stack_reconstruction_error": float(
            np.max(np.abs(apple_reconstruction - apple))
        ),
        "orange_stack_reconstruction_error": float(
            np.max(np.abs(orange_reconstruction - orange))
        ),
    }


def generate_multiresolution_blend() -> dict[str, float | int]:
    output = ASSETS / "multires-lotus-lake"
    output.mkdir(parents=True, exist_ok=True)

    source = SOURCE_IMAGES / "blending" / "lotus-lake"
    lotus = load_rgb(source / "lotus.jpg")
    lake = load_rgb(source / "lake.jpg")
    naive = load_rgb(source / "naive.jpg")
    mask = load_binary_mask(source / "mask.jpg")
    if lotus.shape != lake.shape or lotus.shape != naive.shape:
        raise ValueError("lotus, lake, and naive images must have equal dimensions")
    if mask.shape != lotus.shape[:2]:
        raise ValueError("mask dimensions must match the input images")

    levels = 5
    base_sigma = 2.0
    lotus_gaussian = gaussian_stack(lotus, levels=levels, base_sigma=base_sigma)
    lake_gaussian = gaussian_stack(lake, levels=levels, base_sigma=base_sigma)
    mask_gaussian = gaussian_stack(mask, levels=levels, base_sigma=base_sigma)
    lotus_laplacian = laplacian_stack(lotus_gaussian)
    lake_laplacian = laplacian_stack(lake_gaussian)
    masked_lotus, masked_lake, blended_levels = blend_laplacian_stacks(
        lotus_laplacian, lake_laplacian, mask_gaussian
    )

    lotus_contribution = np.clip(np.sum(masked_lotus, axis=0), 0.0, 1.0)
    lake_contribution = np.clip(np.sum(masked_lake, axis=0), 0.0, 1.0)
    blended = np.clip(np.sum(blended_levels, axis=0), 0.0, 1.0)

    # Bells & whistles: harmonize only the low-frequency Lotus atmosphere.
    # Statistics come from narrow regions just inside and outside the mask.
    lotus_low = gaussian_blur(lotus, sigma=18.0)
    lake_low = gaussian_blur(lake, sigma=18.0)
    lotus_low_lab = cv2.cvtColor(lotus_low.astype(np.float32), cv2.COLOR_RGB2LAB)
    lake_low_lab = cv2.cvtColor(lake_low.astype(np.float32), cv2.COLOR_RGB2LAB)
    morphology_kernel = np.ones((31, 31), dtype=np.uint8)
    mask_u8 = mask.astype(np.uint8)
    inner_ring = mask_u8 - cv2.erode(mask_u8, morphology_kernel)
    outer_ring = cv2.dilate(mask_u8, morphology_kernel) - mask_u8
    source_samples = lotus_low_lab[inner_ring.astype(bool)]
    target_samples = lake_low_lab[outer_ring.astype(bool)]
    source_mean = np.mean(source_samples, axis=0)
    target_mean = np.mean(target_samples, axis=0)
    source_std = np.std(source_samples, axis=0)
    target_std = np.std(target_samples, axis=0)
    scale = np.clip(target_std / np.maximum(source_std, 1e-6), 0.5, 2.0)
    matched_lab = (lotus_low_lab - source_mean) * scale + target_mean
    strength = np.array([0.35, 0.65, 0.65], dtype=np.float32)
    harmonized_lab = lotus_low_lab + strength * (matched_lab - lotus_low_lab)
    harmonized_lab[..., 0] = np.clip(harmonized_lab[..., 0], 0.0, 100.0)
    harmonized_lab[..., 1:] = np.clip(harmonized_lab[..., 1:], -127.0, 127.0)
    harmonized_low = cv2.cvtColor(harmonized_lab, cv2.COLOR_LAB2RGB).astype(np.float64)
    harmonized_lotus = np.clip(harmonized_low + (lotus - lotus_low), 0.0, 1.0)

    enhanced_levels = 6
    enhanced_lotus_g = gaussian_stack(
        srgb_to_linear(harmonized_lotus), levels=enhanced_levels, base_sigma=2.0
    )
    enhanced_lake_g = gaussian_stack(
        srgb_to_linear(lake), levels=enhanced_levels, base_sigma=2.0
    )
    enhanced_mask_g = gaussian_stack(mask, levels=enhanced_levels, base_sigma=2.0)
    enhanced_lotus_l = laplacian_stack(enhanced_lotus_g)
    enhanced_lake_l = laplacian_stack(enhanced_lake_g)
    _, _, enhanced_blended_l = blend_laplacian_stacks(
        enhanced_lotus_l, enhanced_lake_l, enhanced_mask_g
    )
    enhanced_linear = np.clip(np.sum(enhanced_blended_l, axis=0), 0.0, 1.0)
    enhanced_blend = linear_to_srgb(enhanced_linear)

    # Restore the harmonized Lotus atmosphere in the upper image, then fade
    # smoothly into the mask-based color-aware blend through the middle.
    height = lotus.shape[0]
    vertical_position = np.linspace(0.0, 1.0, height)[:, None, None]
    transition = np.clip((vertical_position - 0.34) / (0.60 - 0.34), 0.0, 1.0)
    upper_weight = 0.5 * (1.0 + np.cos(np.pi * transition))
    final_color_aware_linear = (
        upper_weight * srgb_to_linear(harmonized_lotus)
        + (1.0 - upper_weight) * enhanced_linear
    )
    final_color_aware = linear_to_srgb(
        np.clip(final_color_aware_linear, 0.0, 1.0)
    )

    save_rgb(lotus, output / "lotus.jpg", quality=92)
    save_rgb(lake, output / "lake.jpg", quality=92)
    save_gray(mask, output / "mask.jpg", quality=94)
    save_rgb(naive, output / "naive.jpg", quality=92)

    for level in range(levels):
        if level < levels - 1:
            shared_scale = max(
                float(np.percentile(np.abs(masked_lotus[level]), 99.5)),
                float(np.percentile(np.abs(masked_lake[level]), 99.5)),
                float(np.percentile(np.abs(blended_levels[level]), 99.5)),
                1e-12,
            )
        else:
            shared_scale = None
        save_stack_level(
            masked_lotus[level],
            output / f"lotus-level-{level}.jpg",
            signed_scale=shared_scale,
        )
        save_stack_level(
            masked_lake[level],
            output / f"lake-level-{level}.jpg",
            signed_scale=shared_scale,
        )
        save_stack_level(
            blended_levels[level],
            output / f"combined-level-{level}.jpg",
            signed_scale=shared_scale,
        )

    save_rgb(lotus_contribution, output / "lotus-contribution.jpg", quality=94)
    save_rgb(lake_contribution, output / "lake-contribution.jpg", quality=94)
    save_rgb(blended, output / "multires-blend.jpg", quality=94)
    save_rgb(lotus_low, output / "lotus-low-before.jpg", quality=92)
    save_rgb(harmonized_low, output / "lotus-low-harmonized.jpg", quality=92)
    save_rgb(harmonized_lotus, output / "lotus-harmonized.jpg", quality=92)
    save_rgb(enhanced_blend, output / "color-aware-boundary-blend.jpg", quality=94)
    save_rgb(final_color_aware, output / "color-aware-blend.jpg", quality=94)
    return {
        "multires_levels": levels,
        "multires_mask_white_percent": float(100.0 * np.mean(mask)),
        "multires_naive_mae": float(np.mean(np.abs(naive - blended))),
        "color_aware_levels": enhanced_levels,
        "color_aware_upper_full_percent": 34.0,
        "color_aware_upper_fade_end_percent": 60.0,
    }


def generate_pie_cake_blend() -> dict[str, float | int]:
    """Blend an aligned cake and pie across a multiresolution seam."""
    output = ASSETS / "multires-pie-cake"
    output.mkdir(parents=True, exist_ok=True)

    # cake-cut.jpg is the manually aligned version of the cake. Resize every
    # aligned input to one common portrait canvas before building the stacks.
    source = SOURCE_IMAGES / "blending" / "pie-cake"
    pie_source = load_rgb(source / "pexels-pie.jpg")
    height = 1008
    width = round(height * pie_source.shape[1] / pie_source.shape[0])
    pie = resize_rgb(pie_source, width, height)
    cake_aligned = resize_rgb(load_rgb(source / "cake-cut.jpg"), width, height)
    naive = resize_rgb(load_rgb(source / "naive-pie.jpg"), width, height)
    mask_source = load_binary_mask(source / "mask-pie.jpg")
    mask = cv2.resize(mask_source, (width, height), interpolation=cv2.INTER_NEAREST)

    levels = 6
    base_sigma = 2.0
    cake_g = gaussian_stack(
        srgb_to_linear(cake_aligned), levels=levels, base_sigma=base_sigma
    )
    pie_g = gaussian_stack(
        srgb_to_linear(pie), levels=levels, base_sigma=base_sigma
    )
    mask_g = gaussian_stack(mask, levels=levels, base_sigma=base_sigma)
    _, _, blended_l = blend_laplacian_stacks(
        laplacian_stack(cake_g), laplacian_stack(pie_g), mask_g
    )
    blended = linear_to_srgb(
        np.clip(np.sum(blended_l, axis=0), 0.0, 1.0)
    )

    cake_original = resize_long_edge(load_rgb(source / "pexels-cake.jpg"), long_edge=1008)
    save_rgb(cake_original, output / "cake.jpg", quality=92)
    save_rgb(pie, output / "pie.jpg", quality=92)
    save_rgb(cake_aligned, output / "cake-aligned.jpg", quality=92)
    save_gray(mask, output / "mask.jpg", quality=94)
    save_rgb(naive, output / "naive.jpg", quality=92)
    save_rgb(blended, output / "pie-cake-blend.jpg", quality=94)

    return {
        "pie_cake_levels": levels,
        "pie_cake_mask_white_percent": float(100.0 * np.mean(mask)),
        "pie_cake_naive_mae": float(np.mean(np.abs(naive - blended))),
    }


def generate_man_blend() -> dict[str, float | int]:
    """Blend two aligned portraits across a diagonal multiresolution seam."""
    output = ASSETS / "multires-man"
    output.mkdir(parents=True, exist_ok=True)

    source = SOURCE_IMAGES / "blending" / "man"
    left_source = load_rgb(source / "man-left.jpg")
    width = 1100
    height = round(width * left_source.shape[0] / left_source.shape[1])
    left = resize_rgb(left_source, width, height)
    right = resize_rgb(load_rgb(source / "man-right.jpg"), width, height)
    naive = resize_rgb(load_rgb(source / "naive-man.jpg"), width, height)
    mask_source = load_binary_mask(source / "mask-man.jpg")
    mask = cv2.resize(mask_source, (width, height), interpolation=cv2.INTER_NEAREST)

    levels = 6
    base_sigma = 2.0
    right_g = gaussian_stack(
        srgb_to_linear(right), levels=levels, base_sigma=base_sigma
    )
    left_g = gaussian_stack(
        srgb_to_linear(left), levels=levels, base_sigma=base_sigma
    )
    mask_g = gaussian_stack(mask, levels=levels, base_sigma=base_sigma)
    _, _, blended_l = blend_laplacian_stacks(
        laplacian_stack(right_g), laplacian_stack(left_g), mask_g
    )
    blended = linear_to_srgb(
        np.clip(np.sum(blended_l, axis=0), 0.0, 1.0)
    )

    save_rgb(left, output / "man-left.jpg", quality=92)
    save_rgb(right, output / "man-right.jpg", quality=92)
    save_gray(mask, output / "mask.jpg", quality=94)
    save_rgb(naive, output / "naive.jpg", quality=92)
    save_rgb(blended, output / "man-blend.jpg", quality=94)

    return {
        "man_blend_levels": levels,
        "man_mask_white_percent": float(100.0 * np.mean(mask)),
        "man_naive_mae": float(np.mean(np.abs(naive - blended))),
    }


def generate_part2_assets() -> dict[str, float | int]:
    metrics = generate_sharpening()
    metrics.update(generate_hybrids())
    metrics.update(generate_stacks())
    metrics.update(generate_multiresolution_blend())
    metrics.update(generate_pie_cake_blend())
    metrics.update(generate_man_blend())
    metrics_path = ASSETS / "part2-metrics.json"
    with metrics_path.open("w", encoding="utf-8") as output:
        json.dump(metrics, output, indent=2)
    return metrics


if __name__ == "__main__":
    print(json.dumps(generate_part2_assets(), indent=2))
