#!/bin/bash

set -x
set -e

./build.sh

export LD_LIBRARY_PATH=/home/tobe/code/tvm/build/:${LD_LIBRARY_PATH}
./test_tvm_runtime_op.py
