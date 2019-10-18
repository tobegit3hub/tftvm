# TF-TVM

## Introduction

Integrate TensorFlow custom op with TVM Runtime.

TVM provides efficient C++ API to deploy optimized op in any devices. This project provides the TensorFlow custom op for TVM Runtime which can be loaded by TensorFlow session.

## Usage

We can load TVM op from dynamic libraries in TensorFlow graph with `tvm_runtime` op.

```
import tensorflow as tf
from tvm_runtime import tvm_runtime

with tf.Session() as sess:
  a = tf.constant([10.1, 20.0, 11.2, -30.3])
  b = tvm_runtime(a, so_path="tvm_addone_dll.so", function_name="addone")
  print(sess.run(b))
```

## Examples

TVM provides the [test_libs](https://github.com/dmlc/tvm/tree/master/apps/howto_deploy) to introduce C++ deploy API.

We can use the example library to integrated TVM with TensorFlow graph. More details in [examples/addone](./examples/addone/).


## Contribution

The implementation of TensorFlow custom op and Python wrapper are in [tvm_runtime_op](./tvm_runtime_op/).

Currently it needs modification of source code to support int or double dtype and GPU inference.
