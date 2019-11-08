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

class Module():

  def __init__(self, lib_path):
    self.lib_path = lib_path

  def func(self, name, output_dtype="", output_shape="", device=""):
    return Func(self.lib_path, name, output_dtype, output_shape, device)

  def __getitem__(self, func_name):
    return self.func(func_name)


class Func():

  def __init__(self, lib_path, func_name, output_dtype, output_shape, device):
    self.lib_path = lib_path
    self.func_name = func_name
    self.output_dtype = output_dtype
    self.output_shape = output_shape
    self.device = device

    from tensorflow.python.framework import load_library
    tvm_dso_op = load_library.load_op_library('tvm_dso_op.so')
    self.tvm_dso_op  = tvm_dso_op.tvm_dso_op
    
  def apply(self, *params):
    return self.tvm_dso_op(params, lib_path=self.lib_path, func_name=self.func_name, output_dtype=self.output_dtype, output_shape=self.output_shape, device=self.device)

  def __call__(self, *params):
    return self.apply(params)
