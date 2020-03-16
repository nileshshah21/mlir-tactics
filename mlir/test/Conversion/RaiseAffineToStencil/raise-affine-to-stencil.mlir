// RUN: mlir-opt -raise-affine-to-stencil --debug %s | FileCheck %s

func @jacobi1d(%A: memref<1024xf32>, %B: memref<1024xf32>) {
  %cf = constant 0.333333e+00 : f32 
  // CHECK: constant
  // CHECK-NEXT: return
  affine.for %t = 0 to 10 {
    affine.for %i = 1 to 1023 {
      %1 = affine.load %A[%i + 1] : memref<1024xf32>
      %2 = affine.load %A[%i] : memref<1024xf32>
      %3 = affine.load %A[%i - 1] : memref<1024xf32>
      %4 = addf %1, %2 : f32
      %5 = addf %4, %3 : f32
      %6 = mulf %5, %cf : f32
      affine.store %6, %B[%i] : memref<1024xf32>
    }
  }
  return 
}
