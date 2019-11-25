#!/usr/bin/env python

# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

import tvm
import os

def main():
  n = tvm.var("n")
  A = tvm.placeholder((n,), name='A')
  B = tvm.placeholder((n,), name='B')
  C = tvm.compute(A.shape, lambda i: A[i] + B[i], name='C')
  s = tvm.create_schedule(C.op)
  fadd_dylib = tvm.build(s, [A, B, C], "llvm", name="vector_add")
  fadd_dylib.export_library("tvm_add_dll.so")
  
  bx, tx = s[C].split(C.op.axis[0], factor=64)
  s[C].bind(bx, tvm.thread_axis("blockIdx.x"))
  s[C].bind(tx, tvm.thread_axis("threadIdx.x"))
  fadd_dylib = tvm.build(s, [A, B, C], "cuda", name="vector_add")
  fadd_dylib.export_library("tvm_add_cuda_dll.so")
  

if __name__ == "__main__":
  main()
