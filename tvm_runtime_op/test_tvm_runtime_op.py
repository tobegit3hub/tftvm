#!/usr/bin/env python

import tensorflow as tf

from tvm_runtime_op import tvm_runtime

with tf.Session() as sess:
  output = tvm_runtime(tf.constant([10.1, 20.0, 11.2, -30.3]), so_path="lib/test_addone_dll.so", function_name="addone")
  print(sess.run(output))
