// RUN: mlir-opt -test-tactics-blas --debug %s | FileCheck %s

func @gemm(%A: memref<42x42xf32>, %B: memref<42x42xf32>, %C: memref<42x42xf32>) {
  // CHECK: @matmul_42x42x42
  // CHECK-NEXT: return
  affine.for %i = 0 to 42 {
    affine.for %j = 0 to 42 {
      affine.for %k = 0 to 42 {
        %0 = affine.load %A[%i, %k] : memref<42x42xf32>
        %1 = affine.load %B[%k, %j] : memref<42x42xf32>
        %2 = affine.load %C[%i, %j] : memref<42x42xf32>
        %3 = mulf %0, %1 : f32
        %4 = addf %2, %3 : f32
        affine.store %4, %C[%i, %j] : memref<42x42xf32>
      }
    }
  }
  return
}

func @gemmPosOperandTest(%A : memref<5x3xf32>, %B: memref<3x6xf32>, %C: memref<5x6xf32>) {
  // CHECK: @matmul_5x6x3
  // CHECK-NEXT: return
  affine.for %i = 0 to 5 {
    affine.for %j = 0 to 6 {
      affine.for %k = 0 to 3 {
        %0 = affine.load %A[%i, %k] : memref<5x3xf32>
        %1 = affine.load %B[%k, %j] : memref<3x6xf32>
        %2 = affine.load %C[%i, %j] : memref<5x6xf32>
        %3 = mulf %0, %1 : f32
        %4 = addf %2, %3 : f32
        affine.store %4, %C[%i, %j] : memref<5x6xf32>
      }
    }
  }
  return
}

func @gemmTransposeA(%A: memref<3x5xf32>, %B: memref<3x6xf32>, %C: memref<5x6xf32>) {
  // CHECK: @transpose_3x5
  // CHECK-NEXT: @matmul_5x6x3
  // CHECK-NEXT: return
  affine.for %i = 0 to 5 {
    affine.for %j = 0 to 6 {
      affine.for %k = 0 to 3 {
        %0 = affine.load %A[%k, %i] : memref<3x5xf32>
        %1 = affine.load %B[%k, %j] : memref<3x6xf32>
        %2 = affine.load %C[%i, %j] : memref<5x6xf32>
        %3 = mulf %0, %1 : f32
        %4 = addf %2, %3 : f32
        affine.store %4, %C[%i, %j] : memref<5x6xf32>
      }
    }
  }
  return
}

