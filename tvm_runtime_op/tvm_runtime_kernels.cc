
#include <cstdio>
#include <dlpack/dlpack.h>
#include <tvm/runtime/module.h>
#include <tvm/runtime/registry.h>
#include <tvm/runtime/packed_func.h>

#include "tensorflow/core/framework/op_kernel.h"

using namespace tensorflow;


class ZeroOutOp : public OpKernel {
 public:
  explicit ZeroOutOp(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {

    // Grab the input tensor
    const Tensor& input_tensor = context->input(0);
    auto input = input_tensor.flat<int32>();





  tvm::runtime::Module mod_dylib =
      tvm::runtime::Module::LoadFromFile("lib/test_addone_dll.so");
  LOG(INFO) << "Verify dynamic loading from test_addone_dll.so";

  auto fname = "addone";
  auto mod = mod_dylib;

  tvm::runtime::PackedFunc f = mod.GetFunction(fname);
  CHECK(f != nullptr);

  DLTensor* x;
  DLTensor* y;
  int ndim = 1;
  //int dtype_code = kDLFloat;
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


  /*
  for (int i = 0; i < shape[0]; ++i) {
    static_cast<float*>(x->data)[i] = i;
  }
  */
  // TODO: Change to get data from TensoFlow custom op context
  /*
  for (int i = 0; i < shape[0]; ++i) {
    static_cast<int32*>(x->data)[i] = i;
  }
  */


  LOG(INFO) << "XXX work0";
  LOG(INFO) << input;
  LOG(INFO) << "XXX work1";
  x->data = const_cast<int32*>(input.data());
  //x->data = static_cast<int32*>(input);


  const int input_size = input.size();

  /*
  for (int i = 0; i < input_size; ++i) {
    static_cast<int32*>(x->data)[i] = input(i);
    //static_cast<int32*>(x->data)[i] = i;
    //;static_cast<int32*>(x->data)[i] = static_cast<int32>(input(i));
    LOG(INFO) << input(i);
  }
  */

  LOG(INFO) << "XXX work2";


  f(x, y);

  /*
  for (int i = 0; i < i3nput_size; ++i) {
    CHECK_EQ(static_cast<int32*>(y->data)[i], i + 1);
  }
  LOG(INFO) << "Finish verification...";
  */

  LOG(INFO) << y->data;
  LOG(INFO) << "XXX work3";

  for (int i=0; i< input_size; ++i) {
    LOG(INFO) << static_cast<int32*>(y->data)[i];
  }
  LOG(INFO) << "XXX work4";




    // Create an output tensor
    Tensor* output_tensor = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, input_tensor.shape(),
                                                     &output_tensor));
    auto output_flat = output_tensor->flat<int32>();

    // Set all but the first element of the output tensor to 0.
    /*
    const int N = input.size();

    for (int i = 1; i < N; i++) {
      output_flat(i) = 0;
    }]
    */

    /*
    for (int i=0; i<input_size; ++i) {
      output_flat(i) = static_cast<int32*>(y->data)[i];
    }
    */

    //memcpy(y->data, output_flat.data(), input_size*4); 
    memcpy(output_flat.data(), y->data, input_size*4); 
    //output_falt.data()
 
    //output_flat.setData = y->data[i];

    // Preserve the first input value if possible.
    //if (N > 0) output_flat(0) = input(0);
  }
};

REGISTER_KERNEL_BUILDER(Name("ZeroOut").Device(DEVICE_CPU), ZeroOutOp);
