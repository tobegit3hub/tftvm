from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.framework import load_library
from tensorflow.python.platform import resource_loader

tvm_runtime_ops = load_library.load_op_library('tvm_runtime_op.so')
tvm_runtime = tvm_runtime_ops.tvm_runtime
