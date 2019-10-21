#!/bin/bash

set -x
set -e

export LD_LIBRARY_PATH=${TVM_HOME}/build/:${LD_LIBRARY_PATH}
./test_tvm_runtime_op.py
