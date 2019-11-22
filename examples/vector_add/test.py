#!/usr/bin/env python

# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

import tensorflow as tf
import sys

from tvm.contrib import tf_op
# import tf_op


def main():
  module = tf_op.Module("tvm_add_dll.so")
  
  left = tf.placeholder("float32", shape=[4])
  right = tf.placeholder("float32", shape=[4])
  
  feed_dict = {
    left: [1.0, 2.0, 3.0, 4.0],
    right: [5.0, 6.0, 7.0, 8.0]
  }  

  # specify output shape with various styles, output type default to float
  # (1) via static dimensions 
  add1 = module.func("vector_add", output_shape=[4], output_dtype="float")
  # (2) via shape tensor
  add2 = module.func("vector_add", output_shape=tf.shape(left), output_dtype="float")
  # (3) via dimension tensor list
  add3 = module.func("vector_add", output_shape=[tf.shape(left)[0]], output_dtype="float")

  with tf.Session() as sess:
    
    with tf.device("/cpu:0"):
      print(sess.run(add1(left, right), feed_dict))
      print(sess.run(add2(left, right), feed_dict))
      print(sess.run(add3(left, right), feed_dict))

    with tf.device("/gpu:0"):
      add_gpu = tf_op.Module("tvm_add_cuda_dll.so").func("vector_add")
      print(sess.run(add_gpu(left, right), feed_dict))


if __name__ == "__main__":
  main()

