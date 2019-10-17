#!/bin/bash

set -x
set -e

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/tobe/code/tftvm/tensorflow_c_application/lib/

./print_tf_version

