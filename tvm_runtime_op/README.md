# TVM Runtime Op

Firstly, we can implement the op with TVM stack and export the dynamic library.

```
# Define TVM op
n = tvm.var("n")
A = tvm.placeholder((n,), name='A', dtype="int32")
B = tvm.compute(A.shape, lambda *i: A(*i) + 1, name='B')
s = tvm.create_schedule(B.op)

# Export dynamic library
fadd_dylib = tvm.build(s, [A, B], "llvm", name="addone")
dylib_path = os.path.join(base_path, "test_addone_dll.so")
fadd_dylib.export_library(dylib_path)
```

Then register new TensorFlow custom op to get tensor data and run with TVM Runtime API.

```
// Load TVM op
tvm::runtime::Module mod_dylib = tvm::runtime::Module::LoadFromFile("lib/test_addone_dll.so");
const string fname = "addone";
tvm::runtime::PackedFunc f = mod_dylib.GetFunction(fname);

DLTensor* x;
DLTensor* y;
int ndim = 1;
int dtype_code = kDLInt;
int dtype_bits = 32;
int dtype_lanes = 1;
int device_type = kDLCPU;
int device_id = 0;
int64_t shape[1] = {10};
TVMArrayAlloc(shape, ndim, dtype_code, dtype_bits, dtype_lanes, device_type, device_id, &x);
TVMArrayAlloc(shape, ndim, dtype_code, dtype_bits, dtype_lanes, device_type, device_id, &y);

// Get data from TensorFlow
auto input = input_tensor.flat<int32>();
x->data = const_cast<int32*>(input.data());
const int input_size = input.size();

// Run TVM op
f(x, y);

// Output data to TensorFlow
Tensor* output_tensor = NULL;
OP_REQUIRES_OK(context, context->allocate_output(0, input_tensor.shape(), &output_tensor));
auto output_flat = output_tensor->flat<int32>();
memcpy(output_flat.data(), y->data, input_size*4); 
```

Finally, we can use the TVM op like normal TensorFlow op and run by TensorFlow Session.

```
_tvm_runtime_ops = load_library.load_op_library(resource_loader.get_path_to_datafile('tvm_runtime.so'))
tvm_runtime = _tvm_runtime_ops.tvm_runtime

with tf.Session() as sess:
  output = tvm_runtime(tf.constant([10, 20, 11, -30]))
  print(sess.run(output))
```
