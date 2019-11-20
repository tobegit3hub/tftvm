/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 * 
 *   http://www.apache.org/licenses/LICENSE-2.0
 * 
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

#include <cstdio>
#include <cuda_runtime.h>
#include <dlpack/dlpack.h>
#include <tvm/runtime/module.h>
#include <tvm/runtime/device_api.h>
#include <tvm/runtime/registry.h>
#include <tvm/runtime/packed_func.h>
#include "tensorflow/core/framework/op_kernel.h"


using namespace tensorflow;

typedef Eigen::ThreadPoolDevice CPUDevice;
typedef Eigen::GpuDevice GPUDevice;

template <typename DEVICE_TYPE>
class TVMDSOOpTrait;


class TensorAsBuf {
  public:
    Tensor inline_tensor;
    Tensor* tensor;

    size_t size;
    size_t offset;

    int device_type;

    char* origin_buf; 
    char* buf;

    void CopyToOrigin() {
        if (buf == origin_buf) {
            return;
        }
        if (device_type == kDLCPU) {
            memcpy(origin_buf, buf, size); 
        } else {
            cudaMemcpy(origin_buf, buf, size, cudaMemcpyDeviceToDevice);
        }
    }

    void CopyFromOrigin() {
        if (buf == origin_buf) {
            return;
        }
        if (device_type == kDLCPU) {
            memcpy(buf, origin_buf, size); 
        } else {
            cudaMemcpy(buf, origin_buf, size, cudaMemcpyDeviceToDevice);
        }
    }
};


int get_dlpack_datatype(const Tensor& tf_tensor, DLDataType* res) {
    auto dtype = tf_tensor.dtype();
    if (dtype == DT_FLOAT) {
      res->code = kDLFloat;
      res->bits = 32;
      res->lanes = 1;
    } else {
      return -1;
    }
    return 0;
}


void ensure_alignment(OpKernelContext* ctx, const Tensor& tensor, TensorAsBuf* out) {
    char* buf = (char*) tensor.tensor_data().data();
    out->origin_buf = buf;
    out->size = tensor.TotalBytes(); 

    int alignment = 64;
    char* aligned = (char*)(((uint64_t)buf + alignment) & (~ (alignment - 1)));
    if (buf == aligned) {
        out->tensor = const_cast<Tensor*>(&tensor);
        out->buf = buf;
        out->offset = 0;
    } else {
        TensorShape buf_shape;
        int64 dims[1] = { (int64)(tensor.TotalBytes() + alignment) }; 
        TensorShapeUtils::MakeShape(dims, 1, &buf_shape);
        
        out->tensor = &out->inline_tensor;
        ctx->allocate_temp(tensor.dtype(), buf_shape, out->tensor);
        
        buf = (char*)(out->tensor->tensor_data().data());
        char* buf_aligned = (char*)(((uint64_t)buf + alignment) & (~ (alignment - 1)));
        out->buf = buf;
        out->offset = buf_aligned - buf;
    }
}


int make_dltensor(const TensorAsBuf& src, const DLContext& ctx, int64_t* tf_shape, DLTensor* out) {
    DLDataType dlpack_type;
    const Tensor& tensor = *src.tensor;

    int status = get_dlpack_datatype(tensor, &dlpack_type);
    if (status != 0) {
        return status;
    }
    out->ctx = ctx;
    out->ndim = tensor.shape().dims();
    out->shape = tf_shape;
    out->strides = NULL;
    out->byte_offset = 0;
    out->dtype = dlpack_type;    
    out->data = src.buf;
    return 0;
}


template <>
class TVMDSOOpTrait<CPUDevice> {
  public:
    static const int device_type = kDLCPU;

    static int device_id(OpKernelContext* context) {
        return 0;
    }

};


template <>
class TVMDSOOpTrait<GPUDevice> {
  public:
    static const int device_type = kDLGPU;

    static int device_id(OpKernelContext* context) {
        auto device_base = context->device();
        auto gpu_device_info = device_base->tensorflow_gpu_device_info();
        return gpu_device_info->gpu_id;
    }
};


template <typename DEVICE_TYPE>
class TVMDSOOp : public OpKernel {

private:
  tvm::runtime::PackedFunc tvm_func;
  string lib_path;
  string func_name;
  string output_dtype;
  string output_shape;
  string device;

  void initAttributes(OpKernelConstruction* context) {
    context->GetAttr("lib_path", &lib_path);
    context->GetAttr("func_name", &func_name);
    context->GetAttr("output_dtype", &output_dtype);
    context->GetAttr("output_shpae", &output_shape);
    context->GetAttr("device", &device);
  }

 public:
  explicit TVMDSOOp(OpKernelConstruction* context) : OpKernel(context) {

    // Get attr
    initAttributes(context);

    // Load TVM function from dynamic library
    tvm::runtime::Module mod_dylib = tvm::runtime::Module::LoadFromFile(lib_path);
    LOG(INFO) << "Verify dynamic loading from " << lib_path << " device_type=" << TVMDSOOpTrait<DEVICE_TYPE>::device_type;
    tvm_func = mod_dylib.GetFunction(func_name);
    CHECK(tvm_func != nullptr);
  }
  
  void Compute(OpKernelContext* context) override {
    // Grab the input tensor
    auto input_tensor = context->input(0);
    auto input_shape_buf = input_tensor.shape().dim_sizes();
    auto input_shape_ptr = (int64_t*) input_shape_buf.data();

    // Allocate output tensor
    Tensor* output_tensor;
    OP_REQUIRES_OK(context, context->allocate_output(0, input_tensor.shape(), &output_tensor));
    auto output_shape_buf = output_tensor->shape().dim_sizes();    
    auto output_shape_ptr = (int64_t*) output_shape_buf.data();
    
    int device_id = TVMDSOOpTrait<DEVICE_TYPE>::device_id(context);
    int device_type = TVMDSOOpTrait<DEVICE_TYPE>::device_type;

    DLContext dl_ctx = { DLDeviceType(device_type), device_id };

    DLTensor dl_input;
    TensorAsBuf input;
    ensure_alignment(context, input_tensor, &input);

    int status = make_dltensor(input, dl_ctx, input_shape_ptr, &dl_input);
    OP_REQUIRES(context, status == 0, Status(error::INTERNAL, "Fail to create dlpack tensor for input"));

    DLTensor dl_output;
    TensorAsBuf output;
    ensure_alignment(context, *output_tensor, &output);

    status = make_dltensor(output, dl_ctx, output_shape_ptr, &dl_output);
    OP_REQUIRES(context, status == 0, Status(error::INTERNAL, "Fail to create dlpack tensor for output"));

    input.CopyFromOrigin();     

    tvm_func(&dl_input, &dl_output);
   
    output.CopyToOrigin(); 
  }
};




REGISTER_KERNEL_BUILDER(Name("TvmDsoOp").Device(DEVICE_CPU), TVMDSOOp<CPUDevice>);
REGISTER_KERNEL_BUILDER(Name("TvmDsoOp").Device(DEVICE_GPU), TVMDSOOp<GPUDevice>);
