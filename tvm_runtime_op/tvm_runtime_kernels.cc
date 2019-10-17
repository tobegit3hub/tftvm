
#include <cstdio>
#include <dlpack/dlpack.h>
#include <tvm/runtime/module.h>
#include <tvm/runtime/registry.h>
#include <tvm/runtime/packed_func.h>
#include "tensorflow/core/framework/op_kernel.h"

using namespace tensorflow;

class TvmRuntimeOp : public OpKernel {

 public:
  explicit TvmRuntimeOp(OpKernelConstruction* context) : OpKernel(context) {
    // TODO: Need to load dynamic library and specify function
    tvm::runtime::Module mod_dylib =
        tvm::runtime::Module::LoadFromFile("lib/test_addone_dll.so");
    LOG(INFO) << "Verify dynamic loading from test_addone_dll.so";

    const string fname = "addone";
    //tvm::runtime::PackedFunc f = mod_dylib.GetFunction(fname);
    f = mod_dylib.GetFunction(fname);
    CHECK(f != nullptr);
  }
  
  void Compute(OpKernelContext* context) override {
    // Grab the input tensor
    const Tensor& input_tensor = context->input(0);

    // TODO: Need to specify dtype
    auto input = input_tensor.flat<int32>();

    DLTensor* x;
    DLTensor* y;
    int ndim = 1;
    int dtype_code = kDLInt;
    int dtype_bits = 32;
    int dtype_lanes = 1;
    int device_type = kDLCPU;
    int device_id = 0;
    int64_t shape[1] = {10};
    TVMArrayAlloc(shape, ndim, dtype_code, dtype_bits, dtype_lanes,
                  device_type, device_id, &x);
    TVMArrayAlloc(shape, ndim, dtype_code, dtype_bits, dtype_lanes,
                  device_type, device_id, &y);

    // Copy input tensor data to DLPack
    x->data = const_cast<int32*>(input.data());
    const int input_size = input.size();

    f(x, y);

    // Create output tensor from DLPack
    Tensor* output_tensor = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, input_tensor.shape(),
                                                     &output_tensor));
    auto output_flat = output_tensor->flat<int32>();
 
    // TODO: Need to compute data length with dtype
    memcpy(output_flat.data(), y->data, input_size*4); 
   
  }

private:
  tvm::runtime::PackedFunc f;

};

REGISTER_KERNEL_BUILDER(Name("TvmRuntime").Device(DEVICE_CPU), TvmRuntimeOp);
