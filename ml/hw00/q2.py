#!/usr/local/bin/python2.7
import sys
from PIL import Image
import numpy as np

input_file = "Lena.png"

im = Image.open(input_file)
data = np.array(im)
rotated_data = np.flip(data,axis=1)

new_im = Image.fromarray(rotated_data)
new_im.save("ans2.png")
