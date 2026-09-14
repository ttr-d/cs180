# CS194-26 (CS294-26): Project 1 starter Python code

# these are just some suggested libraries
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

# name of the input file
imname = './cathedral.jpg'

# read in the image as grayscale (the glass plate scan is stacked grayscale)
im = cv.imread(imname, cv.IMREAD_GRAYSCALE)

# convert to float in [0,1] (might want to do this later on to save memory)
im = im.astype(np.float32) / 255.0
    
# compute the height of each part (just 1/3 of total)
height = int(np.floor(im.shape[0] / 3.0))

# separate color channels
b = im[:height]
g = im[height: 2*height]
r = im[2*height: 3*height]

# align the images
# functions that might be useful for aligning the images include:
# np.roll, np.sum, sk.transform.rescale (for multiscale)

# ag = align(g, b)
# ar = align(r, b)

# placeholders so the script runs before implementing align()
ag = g
ar = r
# create a color image
im_out = np.dstack([ar, ag, b])

# display the image using matplotlib (expects RGB)
plt.figure(figsize=(8, 8))
plt.imshow(im_out)
plt.title('Colorized')
plt.axis('off')
plt.show()

# prepare for OpenCV saving/display (expects BGR uint8)
out_uint8 = np.clip(im_out * 255.0, 0, 255).astype(np.uint8)
out_bgr = cv.cvtColor(out_uint8, cv.COLOR_RGB2BGR)

# save the image
fname = './out_fname.jpg'
cv.imwrite(fname, out_bgr)

