#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 21 23:11:18 2017

@author: zhanghua
"""
# Import Numpy & Python Imaging Library(PIL)
import numpy as np
from PIL import Image
import os

# Open Images and load it using Image Module
#file_path = '/Users/zhanghua/Downloads/test/'
#file_path = 'C:/Users/yangp/iCloudDrive/test/'
file_path = 'C:/Users/yangp/Desktop/hl2/'
file_list = os.listdir(file_path)
print(file_list)
all_images_list = [file_path+imagefile for imagefile in file_list if 'jpg' in imagefile]
images_len = len(all_images_list)
step_size = 6
images_pos = 0
while images_pos+step_size <= images_len:
    imgs = [ Image.open(i) for i in all_images_list[images_pos: images_pos+4] ]
    # Find the smallest image, and resize the other images to match it
    min_img_shape = sorted( [(np.sum(i.size), i.size ) for i in imgs])[0][1]
    
    # save the horizontally merged images
    #img_merge = np.hstack( (np.asarray( i.resize(min_img_shape,Image.ANTIALIAS) ) for i in imgs ) )
    #img_merge = Image.fromarray( img_merge)
    #img_merge.save( 'terracegarden_h.jpg' )
    
    # Merge the images using vstack and save the Vertially merged images
    img_merge = np.vstack( (np.asarray( i.resize(min_img_shape,Image.ANTIALIAS) ) for i in imgs ) )
    img_merge = Image.fromarray( img_merge)
    img_merge.save( file_path+'v{}.jpg'.format(images_pos))

    images_pos += step_size
    print(images_pos)

print('Mission Complete!')
