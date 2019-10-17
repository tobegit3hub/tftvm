#!/bin/bash

set -x
set -e

#TF_CFLAGS=( $(python -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_compile_flags()))') )
#TF_LFLAGS=( $(python -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_link_flags()))') )

# Change "-L/usr/local/lib/python2.7/site-packages/tensorflow -l:libtensorflow_framework.1.dylib" to "-L/usr/local/lib/python2.7/site-packages/tensorflow"
#g++ -std=c++11 -shared zero_out_kernels.cc zero_out_ops.cc -o zeroout.so -fPIC ${TF_CFLAGS[@]} ${TF_LFLAGS[@]} -O2 -undefined dynamic_lookup
g++ -std=c++11 -shared zero_out_kernels.cc zero_out_ops.cc -o zeroout.so -fPIC -I/usr/local/lib/python2.7/site-packages/tensorflow/include -D_GLIBCXX_USE_CXX11_ABI=0 -L/usr/local/lib/python2.7/site-packages/tensorflow -O2 -undefined dynamic_lookup


