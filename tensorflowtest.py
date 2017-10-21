#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 17 22:20:37 2017

@author: zhanghua
"""

import tensorflow as tf

hello_op = tf.constant('Hello, TensorFlow!')

a = tf.constant(10)
b = tf.constant(32)
compute_op = tf.add(a,b)

with tf.Session() as sess:
    print(sess.run(hello_op))
    print(sess.run(compute_op))