// RUN: mlir-opt -disable-pass-threading=true -test-tactics-blas  %s | FileCheck %s

func @contraction.ab.ac.cd(%A: memref<1024x1024xf32>, %B: memref<1024x1024xf32>, %C: memref<1024x1024xf32>) {
  // CHECK: @matmul_1024x1024x1024
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

func @contraction.ab.acd.dbc(%C: memref<1024x1024xf32>, %A: memref<1024x32x32xf32>, %B: memref<32x1024x32xf32>) {
  // CHECK: @transpose_32x1024x32_to_32x32x1024
  // CHECK: @reshape_1024x32x32_to_1024x1024
  // CHECK: @reshape_32x32x1024_to_1024x1024
  // CHECK: @matmul_1024x1024x1024
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
  // CHECK: @transpose_32x1024x32_to_32x32x1024
  // CHECK: @reshape_32x32x1024_to_1024x1024
  // CHECK: @reshape_32x32x1024_to_1024x1024
  // CHECK: @matmul_1024x1024x1024
  // CHECK: @reshape_1024x1024_to_32x32x1024
  // CHECK: @transpose_32x32x1024_to_32x1024x32
  // CHECK-NEXT: return
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

func @contraction.abc.ad.bdc(%C: memref<1024x32x32xf32>, %A: memref<1024x1024xf32>, %B: memref<32x1024x32xf32>) {
  // CHECK: @transpose_32x1024x32_to_1024x32x32
  // CHECK: @reshape_1024x32x32_to_1024x1024
  // CHECK: @reshape_1024x32x32_to_1024x1024
  // CHECK: @matmul_1024x1024x1024
  // CHECK: @reshape_1024x1024_to_1024x32x32
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

func @contraction.ab.cad.dcb(%C: memref<1024x1024xf32>, %A: memref<32x1024x32xf32>, %B: memref<32x32x1024xf32>) {
  // CHECK: @transpose_32x1024x32_to_1024x32x32
  // CHECK: @transpose_32x32x1024_to_32x32x1024
  // CHECK: @reshape_1024x32x32_to_1024x1024
  // CHECK: @reshape_32x32x1024_to_1024x1024
  // CHECK: @matmul_1024x1024x1024
  affine.for %a = 0 to 1024 {
    affine.for %b = 0 to 1024 {
      affine.for %c = 0 to 32 {
        affine.for %d = 0 to 32{
          %0 = affine.load %A[%c, %a, %d] : memref<32x1024x32xf32>
          %1 = affine.load %B[%d, %c, %b] : memref<32x32x1024xf32>
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

func @contraction.abc.bda.dc(%C: memref<32x32x1024xf32>, %A: memref<32x1024x32xf32>, %B: memref<1024x1024xf32>) {
  // CHECK: @transpose_32x1024x32_to_32x32x1024
  // CHECK: @reshape_32x32x1024_to_1024x1024
  // CHECK: @reshape_32x32x1024_to_1024x1024
  // CHECK: @matmul_1024x1024x1024
  // CHECK: @reshape_1024x1024_to_32x32x1024
  affine.for %a = 0 to 32 {
    affine.for %b = 0 to 32 {
      affine.for %c = 0 to 1024 {
        affine.for %d = 0 to 1024 {
          %0 = affine.load %A[%b, %d, %a] : memref<32x1024x32xf32>
          %1 = affine.load %B[%d, %c] : memref<1024x1024xf32>
          %2 = affine.load %C[%a, %b, %c] : memref<32x32x1024xf32>
          %3 = mulf %0, %1 : f32
          %4 = addf %2, %3 : f32
          affine.store %4, %C[%a, %b, %c] : memref<32x32x1024xf32>
        }
      } 
    }
  }
  return 
}

func @contraction.abcd.aebf.dfce(%C: memref<32x32x32x32xf32>, 
                                 %A: memref<32x32x32x32xf32>, %B: memref<32x32x32x32xf32>) {
  // CHECK: @transpose_32x32x32x32_to_32x32x32x32
  // CHECK: @reshape_1024x32x32_to_1024x1024
  // CHECK: @reshape_32x32x32x32_to_1024x32x32
  // CHECK: @matmul_1024x1024x1024
  // CHECK: @reshape_1024x1024_to_32x32x32x32
  affine.for %a = 0 to 32 {
    affine.for %b = 0 to 32 {
      affine.for %c = 0 to 32 {
        affine.for %d = 0 to 32 {
          affine.for %e = 0 to 32 {
            affine.for %f = 0 to 32 {
              %0 = affine.load %A[%a, %e, %b, %f] : memref<32x32x32x32xf32>
              %1 = affine.load %B[%d, %f, %c, %e] : memref<32x32x32x32xf32>
              %2 = affine.load %C[%a, %b, %c, %d] : memref<32x32x32x32xf32>
              %3 = mulf %0, %1 : f32
              %4 = addf %2, %3 : f32
              affine.store %4, %C[%a, %b, %c, %d] : memref<32x32x32x32xf32>
            }
          }
        }
      }
    }
  }
  return
}


