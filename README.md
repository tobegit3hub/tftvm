# TF-TVM

## Introduction

Integrate TensorFlow custom op with TVM Runtime.

TVM provides the C++ API to deploy optimized op in any devices. We can wrap TVM Runtime as TensorFlow custom op which can be loaded by TensorFlow graph and session.

##  Usage

1. Implement the TVM op in Python and export the dynamic library.

```
cd ./examples/addone/

./prepare_addone_lib.py
```

This will generate the file `tvm_addone_dll.so`.

2. Load TVM op with TensorFlow custom op.

Download the `tvm_runtime_op.so` and `tvm_runtime.py` for your OS. If you want to build from scratch, make sure `tesnorflow` is installed and `TVM_HOME` is set.

```
cd ../../tvm_runtime_op/

./build.sh
```

This will generate the file `tvm_runtime_op.so`.

3. Test with TensorFlow Python script.

```
cd ../examples/addone/

cp ../../tvm_runtime_op/tvm_runtime_op.so ./
cp ../../tvm_runtime_op/tvm_runtime.py ./

export LD_LIBRARY_PATH=${TVM_HOME}/build/:${LD_LIBRARY_PATH}
./test_addone.py
```

## Contribution

The implementation of TensorFlow custom op and Python wrapper are in [tvm_runtime_op](./tvm_runtime_op/).

Currently it needs to modify source code to support int or double dtype and GPU inference.
