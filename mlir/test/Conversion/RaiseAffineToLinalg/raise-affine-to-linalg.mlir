// RUN: mlir-opt -raise-affine-to-linalg --debug %s | FileCheck %s

func @gemm(%arg0: memref<42x42xf32>, %arg1: memref<42x42xf32>, %arg2: memref<42x42xf32>) {
  // CHECK: linalg.generic
  affine.for %arg3 = 0 to 42 {
    affine.for %arg4 = 0 to 42 {
      affine.for %arg5 = 0 to 42 {
        %0 = affine.load %arg0[%arg3, %arg5] : memref<42x42xf32>
        %1 = affine.load %arg1[%arg5, %arg4] : memref<42x42xf32>
        %2 = affine.load %arg2[%arg3, %arg4] : memref<42x42xf32>
        %3 = mulf %0, %1 : f32
        %4 = addf %2, %3 : f32
        affine.store %4, %arg2[%arg3, %arg4] : memref<42x42xf32>
      }
    }
  }
  return
}

func @contraction(%C: memref<42x42x42xf32>, %A: memref<42x42xf32>, %B: memref<42x42x42xf32>) {
  // CHECK: linalg.reshape
  // CHECK-NEXT: linalg.reshape
  // CHECK-NEXT: linalg.generic
  affine.for %i = 0 to 42 {
    affine.for %j = 0 to 42 {
      affine.for %k = 0 to 42 {
        affine.for %l = 0 to 42 {
          %0 = affine.load %A[%i, %l] : memref<42x42xf32>
          %1 = affine.load %B[%l, %j, %k] : memref<42x42x42xf32>
          %3 = mulf %0, %1 : f32
          affine.store %3, %C[%i, %j, %k] : memref<42x42x42xf32>
        }
      }
    }
  }
  return
}

