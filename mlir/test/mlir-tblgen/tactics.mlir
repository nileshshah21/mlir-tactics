// RUN: mlir-opt -test-tactics-linalg --debug %s | FileCheck %s

func @gemmT(%A: memref<22x42xf32>, %B: memref<22x42xf32>, %C: memref<42x42xf32>) {
  // CHECK: linalg.transpose
  // CHECK-NEXT: linalg.generic
  affine.for %i = 0 to 42 {
    affine.for %j = 0 to 42 {
      affine.for %k = 0 to 42 {
        %0 = affine.load %A[%k, %i] : memref<22x42xf32>
        %1 = affine.load %B[%k, %j] : memref<22x42xf32>
        %2 = affine.load %C[%i, %j] : memref<42x42xf32>
        %3 = mulf %0, %1 : f32
        %4 = addf %2, %3 : f32
        affine.store %4, %C[%i, %j] : memref<42x42xf32>
      }
    }
  }
  return
}

func @gemm(%A: memref<22x42xf32>, %B: memref<22x42xf32>, %C: memref<42x42xf32>) {
  // CHECK: linalg.generic
  affine.for %i = 0 to 42 {
    affine.for %j = 0 to 42 {
      affine.for %k = 0 to 42 {
        %0 = affine.load %A[%i, %k] : memref<22x42xf32>
        %1 = affine.load %B[%k, %j] : memref<22x42xf32>
        %2 = affine.load %C[%i, %j] : memref<42x42xf32>
        %3 = mulf %0, %1 : f32
        %4 = addf %2, %3 : f32
        affine.store %4, %C[%i, %j] : memref<42x42xf32>
      }
    }
  }
  return
}
