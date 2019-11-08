#!/usr/bin/env python

import tensorflow as tf
from tensorflow.contrib import graph_editor as ge
from tvm.contrib import tf_op

graph = tf.Graph()

with graph.as_default():
  input_op = tf.constant([1.0, 2.0, -1.0])
  tf_addone = tf.add(input_op, 1.0)
  output_op = tf_addone * 100

  tvm_addone = tf_op.Module("tvm_addone_dll.so")["addone"](input_op)
  new_output_op = ge.graph_replace(output_op, {tf_addone: tvm_addone})

with tf.Session(graph=graph) as sess:
  print(sess.run(output_op))
  print(sess.run(new_output_op))
