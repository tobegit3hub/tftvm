#!/usr/bin/env python

import tensorflow as tf
from tensorflow.python.framework import load_library
from tensorflow.python.platform import resource_loader

_zero_out_ops = load_library.load_op_library(
            #resource_loader.get_path_to_datafile('_zero_out_ops.so'))
            #@resource_loader.get_path_to_datafile('zero_out.so'))
            resource_loader.get_path_to_datafile('tvm_runtime.so'))
            #resource_loader.get_path_to_datafile('./tensorflow_zero_out/python/ops/_zero_out_ops.so'))
zero_out = _zero_out_ops.zero_out

#import ipdb;ipdb.set_trace()

t = zero_out(tf.constant([10, 20, 11, -30]))

sess = tf.Session()

t_value = sess.run(t)
print(t_value)

