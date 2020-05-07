// RUN: mlir-opt -mlir-disable-threading -test-tactics-linalg --debug %s | FileCheck %s

func @gemmT(%A: memref<22x42xf32>, %B: memref<22x42xf32>, %C: memref<42x42xf32>) {
  // CHECK: linalg.generic
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

func @contraction.ab.acd.dbc(%C: memref<1024x1024xf32>, %A: memref<1024x32x32xf32>, %B: memref<32x1024x32xf32>) {
  // CHECK: linalg.transpose %{{.*}} (d0, d1, d2) -> (d2, d0, d1) : memref<32x1024x32xf32>
  // CHECK: linalg.reshape %{{.*}} [#{{.*}}, #{{.*}}] : memref<1024x32x32xf32> into memref<1024x1024xf32>
  // CHECK: linalg.reshape %{{.*}} [#{{.*}}, #{{.*}}] : memref<32x32x1024xf32, #{{.*}}> into memref<1024x1024xf32, #{{.*}}>
  // CHECK: linalg.generic
  affine.for %a = 0 to 1024 {
    affine.for %b = 0 to 1024 {
      affine.for %c = 0 to 32 {
        affine.for %d = 0 to 32 {
          %0 = affine.load %A[%a, %c, %d] : memref<1024x32x32xf32>
          %1 = affine.load %B[%d, %b, %c] : memref<32x1024x32xf32>
          %2 = affine.load %C[%a, %b] : memref<1024x1024xf32>
          %3 = mulf %0, %1 : f32
          %4 = addf %2, %3 : f32
          affine.store %4, %C[%a, %b] : memref<1024x1024xf32>
        }
      }
    }
  }
  return
}

func @contraction.abc.acd.db(%C: memref<32x1024x32xf32>, %A: memref<32x32x1024xf32>, %B: memref<1024x1024xf32>) {
  // CHECK: linalg.transpose %{{.*}} (d0, d1, d2) -> (d0, d2, d1) : memref<32x1024x32xf32>
  // CHECK: linalg.reshape %{{.*}} [#{{.*}}, #{{.*}}] : memref<32x32x1024xf32, #{{.*}}> into memref<?x1024xf32, #{{.*}}>
  // CHECK: linalg.reshape %{{.*}} [#{{.*}}, #{{.*}}] : memref<32x32x1024xf32> into memref<1024x1024xf32>
  // CHECK: linalg.generic
  // CHECK: linalg.reshape %{{.*}} [#{{.*}}, #{{.*}}] : memref<?x1024xf32, #{{.*}}> into memref<32x32x1024xf32, #{{.*}}>
  // CHECK: linalg.transpose %{{.*}} (d0, d1, d2) -> (d0, d2, d1) : memref<32x32x1024xf32, #{{.*}}>
  affine.for %a = 0 to 32 {
    affine.for %b = 0 to 1024 {
      affine.for %c = 0 to 32 {
        affine.for %d = 0 to 1024 {
          %0 = affine.load %A[%a, %c, %d] : memref<32x32x1024xf32>
          %1 = affine.load %B[%d, %b] : memref<1024x1024xf32>
          %2 = affine.load %C[%a, %b, %c] : memref<32x1024x32xf32>
          %3 = mulf %0, %1 : f32
          %4 = addf %2, %3 : f32
          affine.store %4, %C[%a, %b, %c] : memref<32x1024x32xf32>
        }
      }
    }
  }
  return
}

func @mvt(%x1: memref<1024xf32>, %y1: memref<1024xf32>, %A: memref<1024x1024xf32>,
          %x2: memref<1024xf32>, %y2: memref<1024xf32>) {
  // CHECK: linalg.matvec(%{{.*}}, %{{.*}}, %{{.*}}) : memref<1024x1024xf32>, memref<1024xf32>, memref<1024xf32> 
  affine.for %i = 0 to 1024 {
    affine.for %j = 0 to 1024 {
      %0 = affine.load %x1[%i] : memref<1024xf32>
      %1 = affine.load %y1[%j] : memref<1024xf32>
      %2 = affine.load %A[%i, %j] : memref<1024x1024xf32>
      %3 = mulf %2, %1 : f32
      %4 = addf %0, %3 : f32
      affine.store %4, %x1[%i] : memref<1024xf32>
    }
  } 
  return
}

func @contraction.abc.ad.bdc(%C: memref<1024x32x32xf32>, %A: memref<1024x1024xf32>, %B: memref<32x1024x32xf32>) {
  affine.for %a = 0 to 1024 {
    affine.for %b = 0 to 32 {
      affine.for %c = 0 to 32 {
        affine.for %d = 0 to 1024 {
          %0 = affine.load %A[%a, %d] : memref<1024x1024xf32>
          %1 = affine.load %B[%b, %d, %c] : memref<32x1024x32xf32>
          %2 = affine.load %C[%a, %b, %c] : memref<1024x32x32xf32>
          %3 = mulf %0, %1 : f32
          %4 = addf %2, %3 : f32
          affine.store %4, %C[%a, %b, %c] : memref<1024x32x32xf32>
        }
      }
    }
  }
  return
}
