#!/usr/bin/env python

import tensorflow as tf
from tvm_runtime import tvm_runtime

with tf.Session() as sess:
  a = tf.constant([10.1, 20.0, 11.2, -30.3])
  b = tvm_runtime(a, so_path="tvm_addone_dll.so", function_name="addone")
  print(sess.run(b))
