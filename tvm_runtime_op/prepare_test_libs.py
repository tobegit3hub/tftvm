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
"""Script to prepare test_addone.so"""
import tvm
import os


def prepare_test_libs(base_path):
    n = tvm.var("n")
    A = tvm.placeholder((n,), name='A')
    #A = tvm.placeholder((n,), name='A', dtype="int32")
    B = tvm.compute(A.shape, lambda *i: A(*i) + 1, name='B')
    s = tvm.create_schedule(B.op)
    # Compile library as dynamic library
    fadd_dylib = tvm.build(s, [A, B], "llvm", name="addone")
    dylib_path = os.path.join(base_path, "test_addone_dll.so")
    fadd_dylib.export_library(dylib_path)

    bx, tx = s[B].split(B.op.axis[0], factor=64)
    s[B].bind(bx, tvm.thread_axis("blockIdx.x"))
    s[B].bind(tx, tvm.thread_axis("threadIdx.x"))
    fadd_dylib = tvm.build(s, [A, B], "cuda", name="addone")
    dylib_path = os.path.join(base_path, "test_addone_cuda_dll.so")
    fadd_dylib.export_library(dylib_path)
    # Compile library in system library mode
    #fadd_syslib = tvm.build(s, [A, B], "llvm --system-lib", name="addonesys")
    #syslib_path = os.path.join(base_path, "test_addone_sys.o")
    #fadd_syslib.save(syslib_path)


if __name__ == "__main__":
    curr_path = os.path.dirname(os.path.abspath(os.path.expanduser(__file__)))
    prepare_test_libs(os.path.join(curr_path, "./lib"))
