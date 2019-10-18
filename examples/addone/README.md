# Example: AddOne

1. Implement the TVM op in Python and export the dynamic library. This will generate the file `tvm_addone_dll.so`.

```
git clone https://github.com/tobegit3hub/tftvm

cd ./tftvm/examples/addone/

./prepare_addone_lib.py
```

2. Load TVM op with TensorFlow custom op.

Download the `tvm_runtime_op.so` and `tvm_runtime.py` for your OS. If you want to build from scratch, make sure `tesnorflow` is installed and `TVM_HOME` is set. This will generate the file `tvm_runtime_op.so`.

```
cd ../../tvm_runtime_op/

./build.sh
```

3. Test with TensorFlow Python script.

```
cd ../examples/addone/

cp ../../tvm_runtime_op/tvm_runtime_op.so ./
cp ../../tvm_runtime_op/tvm_runtime.py ./

export LD_LIBRARY_PATH=${TVM_HOME}/build/:${LD_LIBRARY_PATH}
./test_addone.py
```