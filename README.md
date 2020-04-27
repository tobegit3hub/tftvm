# TF-TVM

Deprecated notice: This project has been merged into [tvm](https://github.com/apache/incubator-tvm) and please compile with `USE_TF_TVMDSOOP=ON`.

## Introduction

This project enables TensorFlow users to run TVM-optimized operators without effort. 

TVM is one of the most popular compile stack for graph and operator optmization. We can embed TVM in TensorFlow graph to leverage the usability of TensorFlow and extensibility of TVM. The [RFC](https://discuss.tvm.ai/t/rfc-add-tensorflow-custom-op-to-embed-tvm-runtime-in-tensorflow-graph-and-session/4601) is under discussion and this project may migrate to [tvm](https://github.com/apache/incubator-tvm) in the future.

## Installation

Make sure the `TensorFlow` and `TVM` has been installed and setup environment variable of `TVM_HOME`. Notice that the following steps can be skipped once these packages has been migrated to `TVM`.

We can build the `TVMDSOOp` from scratch or download for your OS arch.

```
git clone https://github.com/tobegit3hub/tftvm
cd ./tftvm
sh build.sh
```

Then links the files to existing `TVM` path and set `LD_LIBRARY_PATH`.
```
ln -s $(pwd)/tftvm/python/tf_op/ ${TVM_HOME}/python/tvm/contrib/
ln -s $(pwd)/tftvm/cpp/tvm_dso_op.so ${TVM_HOME}/build/

export LD_LIBRARY_PATH=${TVM_HOME}/build/:${LD_LIBRARY_PATH}
```

## Usage

We can use Python API to load TVM dynamic libraries in TensorFlow graph and session.

```
import tensorflow as tf
from tvm.contrib import tf_op

mod = tf_op.Module("tvm_addone_dll.so")
addone = mod.func("addone", output_shape=[4])

with tf.Session() as sess:
  a = tf.constant([10.1, 20.0, 11.2, -30.3])
  b = addone(a)
  print(sess.run(b))
```

## Examples

[addone](./examples/addone/) is the walk-through example to export TVM libraries and load with TensorFlow.

[vector_add](./examples/vector_add/) is another walk-through example to specify output shape and datatype.

[graph_editor](./graph_editor/addone_example.py) provides the example to edit TensorFlow graph with TVM operator.

## Contribution

Feel free to discuss in [TVM RFC](https://discuss.tvm.ai/t/rfc-add-tensorflow-custom-op-to-embed-tvm-runtime-in-tensorflow-graph-and-session/4601) and any feedback is welcome.
