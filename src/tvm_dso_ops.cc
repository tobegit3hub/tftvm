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

#include "tensorflow/core/framework/op.h"

using namespace tensorflow;

#define REGISTER_TFTVM_OP(n) REGISTER_OP("TvmDsoOp" #n) \
    .Output("output: float") \
    .Attr("lib_path: string") \
    .Attr("func_name: string") \
    .Attr("output_dtype: string") \
    .Attr("output_shape: string") \
    .Attr("device: string")


REGISTER_TFTVM_OP(1).Input("input: float");

REGISTER_TFTVM_OP(2)
    .Input("input1: float")
    .Input("input2: float");

REGISTER_TFTVM_OP(3)
    .Input("input1: float")
    .Input("input2: float")
    .Input("input3: float");

REGISTER_TFTVM_OP(4)
    .Input("input1: float")
    .Input("input2: float")
    .Input("input3: float")
    .Input("input4: float");

REGISTER_TFTVM_OP(5)
    .Input("input1: float")
    .Input("input2: float")
    .Input("input3: float")
    .Input("input4: float")
    .Input("input5: float");

REGISTER_TFTVM_OP(6)
    .Input("input1: float")
    .Input("input2: float")
    .Input("input3: float")
    .Input("input4: float")
    .Input("input5: float")
    .Input("input6: float");

REGISTER_TFTVM_OP(7)
    .Input("input1: float")
    .Input("input2: float")
    .Input("input3: float")
    .Input("input4: float")
    .Input("input5: float")
    .Input("input6: float")
    .Input("input7: float");

REGISTER_TFTVM_OP(8)
    .Input("input1: float")
    .Input("input2: float")
    .Input("input3: float")
    .Input("input4: float")
    .Input("input5: float")
    .Input("input6: float")
    .Input("input7: float")
    .Input("input8: float");
