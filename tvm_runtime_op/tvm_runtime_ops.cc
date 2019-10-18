
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"

using namespace tensorflow;

REGISTER_OP("TvmRuntime")
    .Attr("lib_path: string")
    .Attr("function_name: string")
    .Input("tvm_input: float")
    .Output("tvm_output: float");
    //.SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
    //  c->set_output(0, c->input(0));
    //  return Status::OK();
    //});
