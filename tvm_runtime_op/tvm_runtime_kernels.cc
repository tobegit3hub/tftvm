
#include <cstdio>
#include <cuda_runtime.h>
#include <dlpack/dlpack.h>
#include <tvm/runtime/module.h>
#include <tvm/runtime/registry.h>
#include <tvm/runtime/packed_func.h>
#include "tensorflow/core/framework/op_kernel.h"


using namespace tensorflow;

typedef Eigen::ThreadPoolDevice CPUDevice;
typedef Eigen::GpuDevice GPUDevice;

template <typename DEVICE_TYPE>
class TvmRuntimeOpTrait;

template <>
class TvmRuntimeOpTrait<CPUDevice> {
  public:
    static const int device_type = kDLCPU;
    static int device_id(OpKernelContext* context) {
        return 0;
    }
};

template <>
class TvmRuntimeOpTrait<GPUDevice> {
  public:
    static const int device_type = kDLGPU;
    static int device_id(OpKernelContext* context) {
        auto device_base = context->device();
        auto gpu_device_info = device_base->tensorflow_gpu_device_info();
        return gpu_device_info->gpu_id;
    }
};

template <typename DEVICE_TYPE>
class TvmRuntimeOp : public OpKernel {

private:
  tvm::runtime::PackedFunc tvm_func;
  string lib_path;
  string function_name;

 public:
  explicit TvmRuntimeOp(OpKernelConstruction* context) : OpKernel(context) {

    // Get attr
    context->GetAttr("lib_path", &lib_path);
    context->GetAttr("function_name", &function_name);

    // Load TVM function from dynamic library
    tvm::runtime::Module mod_dylib = tvm::runtime::Module::LoadFromFile(lib_path);
    LOG(INFO) << "Verify dynamic loading from " << lib_path << " device_type=" << TvmRuntimeOpTrait<DEVICE_TYPE>::device_type;
    tvm_func = mod_dylib.GetFunction(function_name);
    CHECK(tvm_func != nullptr);
  }
  
  void Compute(OpKernelContext* context) override {
    // Grab the input tensor
    const Tensor& input_tensor = context->input(0);
    auto input = input_tensor.flat<float>();

    DLTensor* x;
    DLTensor* y;
    int ndim = 1;
    int dtype_code = kDLFloat;
    int dtype_bits = 32;
    int dtype_lanes = 1;
    int device_type = TvmRuntimeOpTrait<DEVICE_TYPE>::device_type;
    int device_id = TvmRuntimeOpTrait<DEVICE_TYPE>::device_id(context);
    int64_t shape[1] = {10};
    TVMArrayAlloc(shape, ndim, dtype_code, dtype_bits, dtype_lanes,
                  device_type, device_id, &x);
    TVMArrayAlloc(shape, ndim, dtype_code, dtype_bits, dtype_lanes,
                  device_type, device_id, &y);

    // Copy input tensor data to DLPack
    x->data = const_cast<float*>(input.data());
    const int input_size = input.size();

    tvm_func(x, y);

    // Create output tensor from DLPack
    Tensor* output_tensor = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, input_tensor.shape(),
                                                     &output_tensor));
    auto output_flat = output_tensor->flat<float>();
 
    // TODO: Use zero-copy instead of memory copy
    if (device_type == kDLCPU) {
        memcpy(output_flat.data(), y->data, input_size*4); 
    } else {
       cudaMemcpy(output_flat.data(), y->data, input_size*4, cudaMemcpyDeviceToDevice);
    }
  }
};


REGISTER_KERNEL_BUILDER(Name("TvmRuntime").Device(DEVICE_CPU), TvmRuntimeOp<CPUDevice>);
REGISTER_KERNEL_BUILDER(Name("TvmRuntime").Device(DEVICE_GPU), TvmRuntimeOp<GPUDevice>);
