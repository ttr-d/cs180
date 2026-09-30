CS 180 PROJECT 2 — REPRODUCING THE RESULTS
===========================================

Supported entry point
---------------------

Run main.py. It executes all experiments from Parts 1.1 through 2.4,
regenerates the webpage figures in ../web/web_assets/, and writes the numerical
measurements to ../web/web_assets/metrics.json and part2-metrics.json.

The program resolves paths relative to its own location, so it may be launched
from any working directory.


Environment
-----------

Python 3.10 or newer is recommended.

From the pj2/code directory, create an environment and install dependencies:

    python -m venv .venv
    .venv\Scripts\activate
    python -m pip install -r requirements.txt

On macOS or Linux, activate the environment with:

    source .venv/bin/activate


Run
---

    python main.py

No command-line arguments or manual intermediate steps are required.


Required input layout
---------------------

main.py expects the original input photographs outside the code directory:

    pj2/
      code/
        main.py
        README.txt
        requirements.txt
        ... helper modules ...
      source_images/
        part1/
        sharpening/
        hybrids/
        stacks/
        blending/
          lotus-lake/
          pie-cake/
          man/
      web/
        index.html
        base.css
        site.css
        site.js
        web_assets/        generated automatically

main.py checks every required input before starting and reports any missing
file by its full expected path. The source_images/unused directory is not used
and is not needed for reproduction.


Where the implementations are located
-------------------------------------

main.py
    The single public entry point. It validates the input set and runs the
    complete workflow.

filter.py
    Part 1 NumPy-only four-loop and two-loop convolution, gradient magnitude,
    manual atan2 approximation, HSV conversion, and orientation rendering.

frequency.py
    Parts 2.1 and 2.2 Gaussian kernels, unsharp masking, alignment, hybrid
    construction, and Fourier-domain measurements.

stacks.py
    Parts 2.3 and 2.4 Gaussian/Laplacian stacks, linear-light conversion, and
    multiresolution blending.

generate_web_assets.py
    Part 1 experiment setup, SciPy validation, timing, figure output, and the
    call into the Part 2 generator.

generate_part2_assets.py
    Part 2 experiment setup, parameter choices, masks, figure output, and
    numerical measurements.

align_image_code.py and hybrid_image_starter.py
    Provided interactive alignment reference code. The automated main.py
    workflow uses the deterministic alignment functions in frequency.py and
    does not require interactive clicks.


Submission note
---------------

Generated web/web_assets/ files are not source code and can be omitted from a
code-only submission because main.py recreates them. However, the original
source_images/ inputs are necessary to reproduce the exact submitted results.
If the submission system or grader already supplies those inputs, they may be
omitted; otherwise they must accompany the code despite not belonging inside
the code/ directory.
