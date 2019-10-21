#!/usr/bin/env python

import tensorflow as tf

from tvm_runtime_ops import tvm_runtime

placeholder = tf.placeholder("float32")
with tf.Session() as sess:
  with tf.device("/gpu:0"):
    output = tvm_runtime(placeholder, lib_path="lib/test_addone_cuda_dll.so", function_name="addone")
    print(sess.run(output, feed_dict={placeholder: [10.1, 20.0, 11.2, -30.3]}))
  
  with tf.device("/cpu:0"): 
    output = tvm_runtime(placeholder, lib_path="lib/test_addone_dll.so", function_name="addone")
    print(sess.run(output, feed_dict={placeholder: [10.1, 20.0, 11.2, -30.3]}))
