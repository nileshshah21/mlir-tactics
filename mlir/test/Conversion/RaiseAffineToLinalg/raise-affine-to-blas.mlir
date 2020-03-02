// RUN: mlir-opt -raise-affine-to-linalg -emit-blas-call --debug %s | FileCheck %s

func @gemm() {

  %arg0 = alloc() : memref<42x42xf32>
  %arg1 = alloc() : memref<42x42xf32>
  %arg2 = alloc() : memref<42x42xf32>

  %cf1 = constant 1.00000e+00 : f32

  linalg.fill(%arg0, %cf1) : memref<42x42xf32>, f32
  linalg.fill(%arg1, %cf1) : memref<42x42xf32>, f32
  linalg.fill(%arg2, %cf1) : memref<42x42xf32>, f32

  // CHECK: @Matmul_42x42x42
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

func @contractionIJK.IL.LJK() {

  %A = alloc() : memref<42x42xf32>
  %B = alloc() : memref<42x42x42xf32>
  %C = alloc() : memref<42x42x42xf32>

  %cf23 = constant 23.00000e+00 : f32
  %cf1 = constant 1.00000e+00 : f32
  
  linalg.fill(%C, %cf1) : memref<42x42x42xf32>, f32
  linalg.fill(%A, %cf23) : memref<42x42xf32>, f32
  linalg.fill(%B, %cf23) : memref<42x42x42xf32>, f32

  // CHECK: call @Reshape_42x42x42_to_42x1764
  // CHECK-NEXT: call @Reshape_42x42x42_to_42x1764
  // CHECK-NEXT: call @Matmul_42x1764x42
  // CHECK-NEXT: call @Reshape_42x1764_to_42x42x42
  affine.for %i = 0 to 42 {
    affine.for %j = 0 to 42 {
      affine.for %k = 0 to 42 {
        affine.for %l = 0 to 42 {
          %0 = affine.load %A[%i, %l] : memref<42x42xf32>
          %1 = affine.load %B[%l, %j, %k] : memref<42x42x42xf32>
          %2 = affine.load %C[%i, %j, %k] : memref<42x42x42xf32>
          %3 = mulf %0, %1 : f32
          %4 = addf %3, %2 : f32
          affine.store %4, %C[%i, %j, %k] : memref<42x42x42xf32>
        }
      }
    }
  }
  return 
}

func @contractionIJK.IL.JLK() {
  
  %A = alloc() : memref<5x8xf32>
  %B = alloc() : memref<6x8x7xf32>
  %C = alloc() : memref<5x6x7xf32>

  %cf23 = constant 23.00000e+00 : f32
  %cf1 = constant 1.00000e+00 : f32

  linalg.fill(%C, %cf1) : memref<5x6x7xf32>, f32
  linalg.fill(%B, %cf23) : memref<6x8x7xf32>, f32
  linalg.fill(%A, %cf23) : memref<5x8xf32>, f32

  // CHECK: call @Reshape_8x6x7_to_8x42
  // CHECK-NEXT: call @Reshape_5x6x7_to_5x42
  // CHECK-NEXT: call @Matmul_5x42x8
  // CHECK-NEXT: call @Reshape_5x42_to_5x6x7
  affine.for %i = 0 to 5 {
    affine.for %j = 0 to 6 {
      affine.for %k = 0 to 7 {
        affine.for %l = 0 to 8 {
          %0 = affine.load %A[%i, %l] : memref<5x8xf32>
          %1 = affine.load %B[%j, %l, %k] : memref<6x8x7xf32>
          %2 = affine.load %C[%i, %j, %k] : memref<5x6x7xf32>
          %3 = mulf %0, %1 : f32
          %4 = addf %3, %2 : f32
          affine.store %4, %C[%i, %j, %k] : memref<5x6x7xf32>
        }
      }
    }
  }
  return
}
