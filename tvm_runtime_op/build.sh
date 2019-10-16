#!/bin/bash

set -x
set -e

TF_CFLAGS=( $(python -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_compile_flags()))') )
TF_LFLAGS=( $(python -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_link_flags()))') )

export LD_LIBRARY_PATH=/home/tobe/code/tvm/build/:${LD_LIBRARY_PATH}

g++ -std=c++11 -shared tvm_runtime_kernels.cc tvm_runtime_ops.cc -o tvm_runtime.so -fPIC ${TF_CFLAGS[@]} ${TF_LFLAGS[@]} -O2 -I/home/tobe/code/tvm/include -I/home/tobe/code/tvm/3rdparty/dmlc-core/include -I/home/tobe/code/tvm/3rdparty/dlpack/include -ltvm_runtime -L/home/tobe/code/tvm/build -ldl -lpthread
