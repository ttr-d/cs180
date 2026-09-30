import matplotlib.pyplot as plt
from pathlib import Path

from align_image_code import align_images

SOURCE_IMAGES = Path(__file__).resolve().parent.parent / "source_images" / "hybrids"

# First load images

# high sf
im1 = plt.imread(SOURCE_IMAGES / "DerekPicture.jpg") / 255.

# low sf
im2 = plt.imread(SOURCE_IMAGES / "nutmeg.jpg") / 255.

# Next align images (this code is provided, but may be improved)
im1_aligned, im2_aligned = align_images(im1, im2)

## You will provide the code below. Sigma1 and sigma2 are arbitrary 
## cutoff values for the high and low frequencies

sigma1 = ...
sigma2 = ...
hybrid = hybrid_image(im1, im2, sigma1, sigma2)

plt.imshow(hybrid)
plt.show()
