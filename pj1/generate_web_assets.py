"""Generate lightweight, reproducible teaching assets for the PJ1 webpage."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

import colorize as c


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "CS180_fa2026_proj1_data"
OUTPUTS = ROOT / "outputs"
ASSETS = ROOT / "web_assets"

SHIFTS: dict[str, tuple[c.Shift, c.Shift]] = {
    "cathedral": ((2, 5), (3, 12)),
    "church": ((4, 25), (-4, 58)),
    "emir": ((24, 49), (40, 107)),
    "harvesters": ((17, 60), (14, 124)),
    "icon": ((17, 42), (23, 90)),
    "ilemselga": ((7, 40), (11, 131)),
    "melons": ((10, 80), (13, 177)),
    "monastery": ((2, -3), (2, 3)),
    "religous_painting": ((5, 30), (6, 70)),
    "self_portrait": ((29, 78), (37, 175)),
    "siren": ((-6, 49), (-24, 96)),
    "three_generations": ((12, 54), (9, 111)),
    "tobolsk": ((2, 3), (3, 6)),
    "wharf": ((-7, 15), (-16, 83)),
}


def _display_array(array: np.ndarray, autocontrast: bool = True) -> np.ndarray:
    result = np.asarray(array, dtype=np.float32).copy()
    if result.ndim == 2:
        result = result[..., None]
    if autocontrast:
        for channel_index in range(result.shape[2]):
            channel = result[..., channel_index]
            low, high = np.percentile(channel, (1, 99))
            if high > low:
                result[..., channel_index] = (channel - low) / (high - low)
    result = np.clip(result * 255, 0, 255).astype(np.uint8)
    if result.shape[2] == 1:
        return result[..., 0]
    return result


def _save(
    array: np.ndarray,
    path: Path,
    max_width: int = 1600,
    quality: int = 90,
    autocontrast: bool = True,
) -> None:
    image = Image.fromarray(_display_array(array, autocontrast=autocontrast))
    if image.width > max_width:
        height = round(image.height * max_width / image.width)
        image = image.resize((max_width, height), Image.Resampling.LANCZOS)
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path, quality=quality, optimize=True)


def _copy_for_web(source: Path, destination: Path, max_width: int = 1500) -> None:
    with Image.open(source) as image:
        image = image.convert("RGB")
        if image.width > max_width:
            height = round(image.height * max_width / image.width)
            image = image.resize((max_width, height), Image.Resampling.LANCZOS)
        destination.parent.mkdir(parents=True, exist_ok=True)
        image.save(destination, quality=88, optimize=True, progressive=True)


def _common_alignment_views(
    blue: np.ndarray,
    green: np.ndarray,
    red: np.ndarray,
    green_shift: c.Shift,
    red_shift: c.Shift,
) -> tuple[np.ndarray, np.ndarray]:
    """Return same-sized unaligned/aligned RGB views for a fair slider."""
    height, width = blue.shape
    shifts = ((0, 0), green_shift, red_shift)
    left = max(dx for dx, _ in shifts)
    right = min(width + dx for dx, _ in shifts)
    top = max(dy for _, dy in shifts)
    bottom = min(height + dy for _, dy in shifts)

    def shifted(channel: np.ndarray, shift: c.Shift) -> np.ndarray:
        dx, dy = shift
        return channel[top - dy : bottom - dy, left - dx : right - dx]

    aligned = np.dstack(
        (shifted(red, red_shift), shifted(green, green_shift), shifted(blue, (0, 0)))
    )
    unaligned = np.dstack(
        (red[top:bottom, left:right], green[top:bottom, left:right],
         blue[top:bottom, left:right])
    )

    crop_top, crop_bottom, crop_left, crop_right = c.detect_crop_bounds(aligned)
    buffer = max(1, round(min(aligned.shape[:2]) * 0.005))
    if crop_top:
        crop_top += buffer
    if crop_bottom < aligned.shape[0]:
        crop_bottom -= buffer
    if crop_left:
        crop_left += buffer
    if crop_right < aligned.shape[1]:
        crop_right -= buffer
    crop = np.s_[crop_top:crop_bottom, crop_left:crop_right]
    return unaligned[crop], aligned[crop]


def _heat_color(value: float) -> tuple[int, int, int]:
    """Dark navy -> blue -> cyan -> cream color ramp."""
    stops = (
        (0.00, (7, 14, 28)),
        (0.35, (20, 55, 92)),
        (0.68, (25, 164, 184)),
        (1.00, (247, 222, 146)),
    )
    for (left_t, left), (right_t, right) in zip(stops, stops[1:]):
        if value <= right_t:
            mix = (value - left_t) / (right_t - left_t)
            return tuple(round(a + mix * (b - a)) for a, b in zip(left, right))
    return stops[-1][1]


def _save_heatmap(
    reference: np.ndarray,
    moving: np.ndarray,
    metric: c.Metric,
    path: Path,
    radius: int = 15,
) -> c.Shift:
    values = np.empty((2 * radius + 1, 2 * radius + 1), dtype=np.float32)
    for row, dy in enumerate(range(-radius, radius + 1)):
        for column, dx in enumerate(range(-radius, radius + 1)):
            ref_patch, mov_patch = c._overlap(reference, moving, (dx, dy), 0.10)
            values[row, column] = c._score(ref_patch, mov_patch, metric)

    best_row, best_column = np.unravel_index(np.argmin(values), values.shape)
    best = (best_column - radius, best_row - radius)
    quality = -values
    low, high = np.percentile(quality, (2, 100))
    normalized = np.clip((quality - low) / max(high - low, 1e-8), 0, 1)

    cell = 16
    pad_left, pad_top, pad_right, pad_bottom = 68, 42, 24, 58
    grid_size = values.shape[0] * cell
    canvas = Image.new(
        "RGB", (pad_left + grid_size + pad_right, pad_top + grid_size + pad_bottom),
        (8, 13, 22),
    )
    draw = ImageDraw.Draw(canvas)
    for row in range(values.shape[0]):
        for column in range(values.shape[1]):
            x0 = pad_left + column * cell
            y0 = pad_top + row * cell
            draw.rectangle(
                (x0, y0, x0 + cell - 1, y0 + cell - 1),
                fill=_heat_color(float(normalized[row, column])),
            )

    marker_x = pad_left + best_column * cell
    marker_y = pad_top + best_row * cell
    draw.rectangle(
        (marker_x - 2, marker_y - 2, marker_x + cell + 1, marker_y + cell + 1),
        outline=(255, 106, 92), width=3,
    )
    draw.text((pad_left, 12), f"BEST  dx={best[0]:+d}, dy={best[1]:+d}", fill=(235, 242, 244))
    draw.text((12, pad_top + grid_size // 2), "dy", fill=(131, 159, 171))
    draw.text((pad_left + grid_size // 2, pad_top + grid_size + 30), "dx", fill=(131, 159, 171))
    for tick in (-15, -10, -5, 0, 5, 10, 15):
        position = (tick + radius) * cell + cell // 2
        draw.text((pad_left + position - 7, pad_top + grid_size + 10), str(tick), fill=(131, 159, 171))
        draw.text((pad_left - 28, pad_top + position - 5), str(tick), fill=(131, 159, 171))
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path, optimize=True)
    return best


def generate_explanation_assets() -> None:
    plate = c.read_grayscale(DATA / "cathedral.jpg")
    blue, green, red = c.split_channels(plate)
    _save(plate, ASSETS / "plate-cathedral.jpg", max_width=720)
    _save(blue, ASSETS / "channel-blue.jpg", max_width=520)
    _save(green, ASSETS / "channel-green.jpg", max_width=520)
    _save(red, ASSETS / "channel-red.jpg", max_width=520)

    unaligned, aligned = _common_alignment_views(
        blue, green, red, *SHIFTS["cathedral"]
    )
    _save(unaligned, ASSETS / "cathedral-unaligned.jpg", max_width=1000)
    _save(aligned, ASSETS / "cathedral-aligned.jpg", max_width=1000)
    ncc_best = _save_heatmap(blue, green, "ncc", ASSETS / "heatmap-ncc.png")
    l2_best = _save_heatmap(blue, green, "l2", ASSETS / "heatmap-l2.png")

    emir_plate = c.read_grayscale(DATA / "emir.tif")
    emir_blue, emir_green, emir_red = c.split_channels(emir_plate)
    level = emir_blue
    index = 0
    while True:
        _save(level, ASSETS / f"pyramid-emir-{index}.jpg", max_width=1200)
        if min(level.shape) <= 400:
            break
        level = c._downsample(level)
        index += 1

    raw_emir = c.compose_rgb(
        emir_blue, emir_green, emir_red, *SHIFTS["emir"], crop_mode="none"
    )
    cropped_emir = c.crop_borders(raw_emir, mode="auto")
    _save(raw_emir, ASSETS / "emir-before-crop.jpg", max_width=1500)
    _save(cropped_emir, ASSETS / "emir-after-crop.jpg", max_width=1500)
    _save(cropped_emir, ASSETS / "hero-emir.jpg", max_width=2200, quality=92)
    print(f"Single-scale heatmaps: NCC {ncc_best}, L2 {l2_best}")


def generate_gallery() -> None:
    gallery = ASSETS / "gallery"
    for stem, shifts in SHIFTS.items():
        source = OUTPUTS / f"{stem}_color.jpg"
        if not source.exists():
            matches = list(DATA.glob(f"{stem}.*"))
            if not matches:
                raise FileNotFoundError(f"No source scan found for {stem}")
            c.colorize(matches[0], source, autocontrast=True)
        _copy_for_web(source, gallery / f"{stem}.jpg")
        print(f"Gallery: {stem:20s} G={shifts[0]} R={shifts[1]}")


def main() -> None:
    generate_explanation_assets()
    generate_gallery()


if __name__ == "__main__":
    main()
