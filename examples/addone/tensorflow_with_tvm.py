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
from tvm.contrib import tf_op

def main():
  mod = tf_op.Module("tvm_addone_dll.so")
  addone = mod["addone"]

  with tf.Session() as sess:
    a = tf.constant([10.1, 20.0, 11.2, -30.3])
    b = addone(a)
    print(sess.run(b))

    with tf.device("/cpu:0"):
      print(sess.run(addone(tf.constant([1.0, 2.0]))))

    with tf.device("/gpu:0"):
      placeholder = tf.placeholder("float32")
      addone_gpu = tf_op.Module("tvm_addone_cuda_dll.so")["addone"]
      print(sess.run(addone_gpu(placeholder), feed_dict={placeholder: [1.0, 2.0]}))
      #print(sess.run(addone_gpu(tf.constant([1.0, 2.0], dtype=tf.float32))))

if __name__ == "__main__":
  main()
