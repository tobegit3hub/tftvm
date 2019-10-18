#!/usr/bin/env python

import tensorflow as tf

from tvm_runtime_op import tvm_runtime

with tf.Session() as sess:
  output = tvm_runtime(tf.constant([10, 20, 11, -30]), so_path="lib/test_addone_dll.so", function_name="addone")
  print(sess.run(output))
