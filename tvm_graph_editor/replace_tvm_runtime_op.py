#!/usr/bin/env python

import tensorflow as tf
from tensorflow.python.framework import load_library
from tensorflow.python.platform import resource_loader
from tensorflow.contrib import graph_editor as ge

_tvm_runtime_ops = load_library.load_op_library(
            resource_loader.get_path_to_datafile('tvm_runtime.so'))
tvm_runtime = _tvm_runtime_ops.tvm_runtime

graph = tf.Graph()

with graph.as_default():
  input_op = tf.constant([1, 2, -1])
  add_one = input_op + 1
  result = add_one + 100

tvm_add_one = tvm_runtime(input_op)
new_result = ge.graph_replace(result, {add_one: tvm_add_one})

with tf.Session(graph=graph) as sess:
  print(sess.run(result))
  print(sess.run(new_result))
