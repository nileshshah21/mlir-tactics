// RUN: mlir-opt -test-tactics %s | FileCheck %s

func @contractionT(%C: memref<5x6x7xf32>, %A: memref<5x8xf32>, %B: memref<6x8x7xf32>) {
  // CHECK-NOT: affine.for
  affine.for %i = 0 to 5 {
    affine.for %j = 0 to 6 {
      affine.for %k = 0 to 7 {
        affine.for %l = 0 to 8 {
          %0 = affine.load %A[%i, %l] : memref<5x8xf32>
          %1 = affine.load %B[%j, %l, %k] : memref<6x8x7xf32>
          %3 = mulf %0, %1 : f32
          affine.store %3, %C[%i, %j, %k] : memref<5x6x7xf32>
        }
      }
    }
  }
  return
}
