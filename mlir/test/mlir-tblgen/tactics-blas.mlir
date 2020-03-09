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
  // CHECK: @transpose_3x5_to_5x3
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

func @contraction(%A: memref<2x5xf32>, %B: memref<3x5x4xf32>, %C: memref<2x3x4xf32>) {
  // CHECK: alloc() : memref<5x3x4xf32>
  // CHECK-NEXT: llvm.mlir.addressof
  // CHECK-NEXT: llvm.mlir.constant
  // CHECK-NEXT: @transpose_3x5x4_to_5x3x4
  // CHECK-NEXT: alloc() : memref<5x12xf32>
  // CHECK-NEXT: @reshape_5x3x4_to_5x12
  // CHECK-NEXT: alloc() : memref<2x12xf32>
  // CHECK-NEXT: @reshape_2x3x4_to_2x12
  // CHECK-NEXT: @matmul_2x12x5
  // CHECK-NEXT: @reshape_2x12_to_2x3x4
  affine.for %i = 0 to 2 {
    affine.for %j = 0 to 3 {
      affine.for %k = 0 to 4 {
        affine.for %l = 0 to 5 {
          %0 = affine.load %A[%i, %l] : memref<2x5xf32>
          %1 = affine.load %B[%j, %l, %k] : memref<3x5x4xf32>
          %2 = affine.load %C[%i, %j, %k] : memref<2x3x4xf32>
          %3 = mulf %0, %1 : f32
          %4 = addf %2, %3 : f32
          affine.store %4, %C[%i, %j, %k] : memref<2x3x4xf32>
        }
      }
    }
  }
  return
}
