#!/bin/bash

set -x
set -e

#TF_CFLAGS=( $(python -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_compile_flags()))') )
#TF_LFLAGS=( $(python -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_link_flags()))') )

#g++ -std=c++11 -shared zero_out_kernels.cc zero_out_ops.cc -o zeroout.so -fPIC ${TF_CFLAGS[@]} ${TF_LFLAGS[@]} -O2 -undefined dynamic_lookup

gcc -I/Users/tobe/temp2/libtensorflow-cpu-darwin-x86_64-1.13.1/include -L/Users/tobe/temp2/libtensorflow-cpu-darwin-x86_64-1.13.1/lib  print_tf_version.c -ltensorflow -o print_tf_version
