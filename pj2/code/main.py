"""Single reproducible entry point for CS 180 Project 2.

Run this file to execute every experiment from Parts 1.1 through 2.4, rebuild
all webpage figures, and recompute the numerical validation metrics.
"""

from __future__ import annotations

from pathlib import Path

from generate_web_assets import main as generate_all_assets


CODE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = CODE_ROOT.parent
SOURCE_IMAGES = PROJECT_ROOT / "source_images"

# Only inputs used by the reproducible pipeline are required.  Generated files
# under web/web_assets/ are deliberately excluded because main.py recreates
# them.
REQUIRED_INPUTS = (
    "part1/selfie.jpg",
    "part1/cameraman.png",
    "sharpening/taj.jpg",
    "hybrids/DerekPicture.jpg",
    "hybrids/nutmeg.jpg",
    "hybrids/iestein.png",
    "hybrids/marilyn.png",
    "hybrids/raccoon.jpg",
    "hybrids/wolf.jpg",
    "stacks/apple.jpeg",
    "stacks/orange.jpeg",
    "blending/lotus-lake/lotus.jpg",
    "blending/lotus-lake/lake.jpg",
    "blending/lotus-lake/mask.jpg",
    "blending/lotus-lake/naive.jpg",
    "blending/pie-cake/pexels-pie.jpg",
    "blending/pie-cake/pexels-cake.jpg",
    "blending/pie-cake/cake-cut.jpg",
    "blending/pie-cake/mask-pie.jpg",
    "blending/pie-cake/naive-pie.jpg",
    "blending/man/man-left.jpg",
    "blending/man/man-right.jpg",
    "blending/man/mask-man.jpg",
    "blending/man/naive-man.jpg",
)


def validate_inputs() -> None:
    """Fail early with a readable list when required source images are absent."""
    missing = [
        SOURCE_IMAGES / relative_path
        for relative_path in REQUIRED_INPUTS
        if not (SOURCE_IMAGES / relative_path).is_file()
    ]
    if missing:
        formatted = "\n".join(f"  - {path}" for path in missing)
        raise FileNotFoundError(
            "The following source images are required to reproduce PJ2:\n"
            f"{formatted}\n"
            "See code/README.txt for the expected directory structure."
        )


def main() -> None:
    """Validate inputs, run Parts 1.1--2.4, and regenerate web/web_assets/."""
    validate_inputs()
    generate_all_assets()


if __name__ == "__main__":
    main()
