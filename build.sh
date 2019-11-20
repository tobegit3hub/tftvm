#!/bin/bash

mkdir -p build
cd build
cmake .. -DTVM_HOME=${TVM_HOME}
make
