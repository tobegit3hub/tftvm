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
sys.path.append("./python/")
#from tvm.contrib import tf_op
import tf_op

def main():
  mod = tf_op.Module("tvm_add_dll.so")
  add = mod["add"]

  with tf.Session() as sess:
    
    with tf.device("/cpu:0"):
      left = tf.placeholder("float32", shape=[2])
      right = tf.placeholder("float32", shape=[2])
      print(sess.run(add(left, right), feed_dict={left: [1.0, 2.0], right: [3.0, 4.0]}))

    with tf.device("/gpu:0"):
      left = tf.placeholder("float32", shape=[2])
      right = tf.placeholder("float32", shape=[2])
      add_gpu = tf_op.Module("tvm_add_cuda_dll.so")["add"]
      print(sess.run(add_gpu(left, right), feed_dict={left: [1.0, 2.0], right: [3.0, 4.0]}))


if __name__ == "__main__":
  main()
