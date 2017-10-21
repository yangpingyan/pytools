#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 21 23:11:18 2017

@author: zhanghua
"""
# Import Numpy & Python Imaging Library(PIL)
import numpy as np
from PIL import Image

# Open Images and load it using Image Module
images_list = ['1.png', '2.png', '3.png']
imgs = [ Image.open(i) for i in images_list ]
print(imgs)
# Find the smallest image, and resize the other images to match it
min_img_shape = sorted( [(np.sum(i.size), i.size ) for i in imgs])[0][1]

# save the horizontally merged images
#img_merge = np.hstack( (np.asarray( i.resize(min_img_shape,Image.ANTIALIAS) ) for i in imgs ) )
#img_merge = Image.fromarray( img_merge)
#img_merge.save( 'terracegarden_h.jpg' )

# Merge the images using vstack and save the Vertially merged images
img_merge = np.vstack( (np.asarray( i.resize(min_img_shape,Image.ANTIALIAS) ) for i in imgs ) )
img_merge = Image.fromarray( img_merge)
img_merge.save( 'terracegarden_v.jpg' )


print('Mission Complete!')
