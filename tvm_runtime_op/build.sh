#!/bin/bash

set -x
set -e

TF_CFLAGS=( $(python -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_compile_flags()))') )
TF_LFLAGS=( $(python -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_link_flags()))') )

g++ -g -std=c++11 -shared tvm_runtime_kernels.cc tvm_runtime_ops.cc -o tvm_runtime_op.so -fPIC ${TF_CFLAGS[@]} ${TF_LFLAGS[@]} -O2 -I${TVM_HOME}/include -I${TVM_HOME}/3rdparty/dmlc-core/include -I${TVM_HOME}/3rdparty/dlpack/include -ltvm_runtime -L${TVM_HOME}/build -ldl -lpthread 
