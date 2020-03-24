// RUN: mlir-opt -disable-pass-threading=true -test-tactics-blas-gpu  %s | FileCheck %s

func @contraction.ab.ac.cd(%A: memref<1024x1024xf32>, %B: memref<1024x1024xf32>, %C: memref<1024x1024xf32>) {
  // CHECK: call @allocateMemoryForDevice
  // CHECK-NEXT: call @allocateMemoryForDevice
  // CHECK-NEXT: call @allocateMemoryForDevice
  // CHECK-NEXT: @createCallCopyFromHostToDevice
  // CHECK-NEXT: @createCallCopyFromHostToDevice
  // CHECK-NEXT: @createCallCopyFromHostToDevice
  // CHECK-NEXT: @createCallToCublasSgemm
  // CHECK-NEXT: @createCallCopyFromDeviceToHost
  // CHECK-NEXT: return
  affine.for %i = 0 to 1024 {
    affine.for %j = 0 to 1024 {
      affine.for %k = 0 to 1024 {
        %0 = affine.load %A[%i, %k] : memref<1024x1024xf32>
        %1 = affine.load %B[%k, %j] : memref<1024x1024xf32>
        %2 = affine.load %C[%i, %j] : memref<1024x1024xf32>
        %3 = mulf %0, %1 : f32
        %4 = addf %2, %3 : f32
        affine.store %4, %C[%i, %j] : memref<1024x1024xf32>
      }
    }
  }
  return
}


