#!/bin/bash

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

set -x
set -e

TF_CFLAGS=( $(python -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_compile_flags()))') )
TF_LFLAGS=( $(python -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_link_flags()))') )

op_kernel_file=tvm_dso_op_kernels.cc
op_register_file=tvm_dso_ops.cc
output_so_file=tvm_dso_op.so

g++ -std=c++11 -shared $op_kernel_file $op_register_file -o $output_so_file -fPIC ${TF_CFLAGS[@]} ${TF_LFLAGS[@]} -O2 -I${TVM_HOME}/include -I${TVM_HOME}/3rdparty/dmlc-core/include -I${TVM_HOME}/3rdparty/dlpack/include  -ldl -lpthread -I/usr/local/cuda/include
