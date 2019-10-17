#!/usr/bin/env python

import tensorflow as tf
from tensorflow.python.framework import load_library
from tensorflow.python.platform import resource_loader

_tvm_runtime_ops = load_library.load_op_library(
            resource_loader.get_path_to_datafile('tvm_runtime.so'))
tvm_runtime = _tvm_runtime_ops.tvm_runtime

with tf.Session() as sess:
  output = tvm_runtime(tf.constant([10, 20, 11, -30]))
  print(sess.run(output))
