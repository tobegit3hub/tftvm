#!/usr/bin/env python

import tvm
import os

def main():
  n = tvm.var("n")
  A = tvm.placeholder((n,), name='A')
  B = tvm.compute(A.shape, lambda *i: A(*i) + 1, name='B')
  s = tvm.create_schedule(B.op)

  fadd_dylib = tvm.build(s, [A, B], "llvm", name="addone")
  fadd_dylib.export_library("tvm_addone_dll.so")

if __name__ == "__main__":
  main()
