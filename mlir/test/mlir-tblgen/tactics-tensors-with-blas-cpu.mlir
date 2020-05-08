// RUN: mlir-opt -mlir-disable-threading -test-tactics-blas-cpu  %s | FileCheck %s

func @checkFunctionName(%A : memref<5x3xf32>, %B: memref<3x6xf32>, %C: memref<5x6xf32>) {
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

// check if we can catch a gemm pattern
// also if we have interleaved operations
// between the loads.
func @contraction.ab.ac.cd.withInterleavedMul(%A: memref<1024x1024xf32>, 
                                              %B: memref<1024x1024xf32>, %C: memref<1024x1024xf32>) {
  // CHECK: @matmul_1024x1024x1024
  // CHECK-NEXT: return
  affine.for %i = 0 to 1024 {
    affine.for %j = 0 to 1024 {
      affine.for %k = 0 to 1024 {
        %0 = affine.load %A[%i, %k] : memref<1024x1024xf32>
        %1 = affine.load %B[%k, %j] : memref<1024x1024xf32>
        %3 = mulf %0, %1 : f32
        %2 = affine.load %C[%i, %j] : memref<1024x1024xf32>
        %4 = addf %2, %3 : f32
        affine.store %4, %C[%i, %j] : memref<1024x1024xf32>
      }
    }
  }
  return
}

func @distributed.3mm(%arg0: memref<800x1000xf32>, 
                      %arg1: memref<1000x900xf32>, 
                      %arg2: memref<900x1200xf32>, 
                      %arg3: memref<1200x1100xf32>, 
                      %arg4: memref<800x900xf32>, 
                      %arg5: memref<900x1100xf32>, 
                      %arg6: memref<800x1100xf32>) {
    // G[i][j] = 0
    affine.for %arg7 = 0 to 800 {
      affine.for %arg8 = 0 to 1100 {
        %cst = constant 0.000000e+00 : f32
        affine.store %cst, %arg6[%arg7, %arg8] : memref<800x1100xf32>
      }
    }
    // F[i][j] = 0
    affine.for %arg7 = 0 to 900 {
      affine.for %arg8 = 0 to 1100 {
        %cst = constant 0.000000e+00 : f32
        affine.store %cst, %arg5[%arg7, %arg8] : memref<900x1100xf32>
      }
    }
    // E[i][j] = 0
    affine.for %arg7 = 0 to 800 {
      affine.for %arg8 = 0 to 900 {
        %cst = constant 0.000000e+00 : f32
        affine.store %cst, %arg4[%arg7, %arg8] : memref<800x900xf32>
      }
    }
    // E[i][j] += A[i][j] * B[k][j]
    // CHECK: @matmul_800x900x1000
    affine.for %arg7 = 0 to 800 {
      affine.for %arg8 = 0 to 900 {
        affine.for %arg9 = 0 to 1000 {
          %0 = affine.load %arg0[%arg7, %arg9] : memref<800x1000xf32>
          %1 = affine.load %arg1[%arg9, %arg8] : memref<1000x900xf32>
          %2 = affine.load %arg4[%arg7, %arg8] : memref<800x900xf32>
          %3 = mulf %0, %1 : f32 
          %4 = addf %2, %3 : f32
          affine.store %4, %arg4[%arg7, %arg8] : memref<800x900xf32>
        }
      }
    }
    // F[i][j] += C[i][k] * D[k][j]
    // CHECK: @matmul_900x1100x1200
    affine.for %arg7 = 0 to 900 {
      affine.for %arg8 = 0 to 1100 {
        affine.for %arg9 = 0 to 1200 {
          %0 = affine.load %arg2[%arg7, %arg9] : memref<900x1200xf32>
          %1 = affine.load %arg3[%arg9, %arg8] : memref<1200x1100xf32>
          %2 = affine.load %arg5[%arg7, %arg8] : memref<900x1100xf32>
          %3 = mulf %0, %1 : f32
          %4 = addf %2, %3 : f32
          affine.store %4, %arg5[%arg7, %arg8] : memref<900x1100xf32>
        }
      }
    }
    // G[i][j] += E[i][k] * F[k][j]
    // CHECK: @matmul_800x1100x900
    affine.for %arg7 = 0 to 800 {
      affine.for %arg8 = 0 to 1100 {
        affine.for %arg9 = 0 to 900 {
          %0 = affine.load %arg4[%arg7, %arg9] : memref<800x900xf32>
          %1 = affine.load %arg5[%arg9, %arg8] : memref<900x1100xf32>
          %2 = affine.load %arg6[%arg7, %arg8] : memref<800x1100xf32>
          %3 = mulf %0, %1 : f32 
          %4 = addf %2, %3 : f32
          affine.store %4, %arg6[%arg7, %arg8] : memref<800x1100xf32>
        }
      }
    }
    return
}

//func @contraction.ab.ca.cd(%A: memref<3x5xf32>, %B: memref<3x6xf32>, %C: memref<5x6xf32>) {
//  [FIXME: expect matmul_5x6x3 but the generated label is 5x6x5 due to the
//  fact that composeFunctionNameForMatmul does not know about transposition.
//  affine.for %i = 0 to 5 {
//    affine.for %j = 0 to 6 {
//      affine.for %k = 0 to 3 {
//        %0 = affine.load %A[%k, %i] : memref<3x5xf32>
//        %1 = affine.load %B[%k, %j] : memref<3x6xf32>
//        %2 = affine.load %C[%i, %j] : memref<5x6xf32>
//        %3 = mulf %0, %1 : f32
//        %4 = addf %2, %3 : f32
//        affine.store %4, %C[%i, %j] : memref<5x6xf32>
//      }
//    }
//  }
//  return
//}

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
  // CHECK: @transpose_32x32x32x32_to_32x32x32x32(%arg1, %{{.*}}, %{{.*}}, %{{.*}})
  // CHECK: @transpose_32x32x32x32_to_32x32x32x32(%arg2, %{{.*}}, %{{.*}}, %{{.*}})
  // CHECK: @reshape_32x32x32x32_to_1024x1024(%arg0, %{{.*}})
  // CHECK: @reshape_32x32x32x32_to_1024x1024(%0, %{{.*}})
  // CHECK: @reshape_32x32x32x32_to_1024x1024(%3, %{{.*}})
  // CHECK: @matmul_1024x1024x1024
  // CHECK: @reshape_1024x1024_to_32x32x32x32({{.*}}, %arg0)
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

func @contraction.abcd.aebf.fdec(%C: memref<32x32x32x32xf32>,
                                 %A: memref<32x32x32x32xf32>, %B: memref<32x32x32x32xf32>) {
  // CHECK: @transpose_32x32x32x32_to_32x32x32x32(%arg1, %{{.*}}, %{{.*}}, %{{.*}})
  // CHECK: @transpose_32x32x32x32_to_32x32x32x32(%arg2, %{{.*}}, %{{.*}}, %{{.*}})
  // CHECK: @reshape_32x32x32x32_to_1024x1024(%arg0, %{{.*}})
  // CHECK: @reshape_32x32x32x32_to_1024x1024(%0, %{{.*}})
  // CHECK: @reshape_32x32x32x32_to_1024x1024(%3, %{{.*}})
  // CHECK: @matmul_1024x1024x1024
  // CHECK: @reshape_1024x1024_to_32x32x32x32({{.*}}, %arg0) 
  affine.for %a = 0 to 32 {
    affine.for %b = 0 to 32 {
      affine.for %c = 0 to 32 {
        affine.for %d = 0 to 32 {
          affine.for %e = 0 to 32 {
            affine.for %f = 0 to 32 {
              %0 = affine.load %A[%a, %e, %b, %f] : memref<32x32x32x32xf32>
              %1 = affine.load %B[%f, %d, %e, %c] : memref<32x32x32x32xf32>
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

func @mvt(%x1: memref<1024xf32>, %y1: memref<1024xf32>, %A: memref<1024x1024xf32>,
          %x2: memref<1024xf32>, %y2: memref<1024xf32>) {

  // CHECK: %cst = constant 1.000000e+00 : f32
  // CHECK: %0 = llvm.mlir.constant(0 : i32) : !llvm.i32
  // CHECK: @matvec_1024x1024x1024(%{{.*}}, %{{.*}}, %{{.*}}, %cst, %cst, %0) 
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

  // CHECK: %1 = llvm.mlir.constant(1 : i32) : !llvm.i32
  // CHECK: @matvec_1024x1024x1024(%{{.*}}, %{{.*}}, %{{.*}}, %cst, %cst, %1)
  affine.for %i = 0 to 1024 {
    affine.for %j = 0 to 1024 {
      %0 = affine.load %x1[%i] : memref<1024xf32>
      %1 = affine.load %y1[%j] : memref<1024xf32>
      %2 = affine.load %A[%j, %i] : memref<1024x1024xf32>
      %3 = mulf %2, %1 : f32
      %4 = addf %0, %3 : f32
      affine.store %4, %x1[%i] : memref<1024xf32>
    }
  }
  return 
}

func @mvtWithConstant(%arg0: memref<2000x2000xf32>,
                      %arg1: f32, %arg2: f32,
                      %arg7: memref<2000xf32>,
                      %arg8: memref<2000xf32>, %arg9: memref<2000xf32>) {

    // CHECK: %cst = constant 1.000000e+00 : f32
    // CHECK: %0 = llvm.mlir.constant(1 : i32) : !llvm.i32
    // CHECK: @matvec_2000x2000x2000(%{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %cst, %0) 
    affine.for %arg11 = 0 to 2000 {
      affine.for %arg12 = 0 to 2000 {
        %0 = affine.load %arg8[%arg11] : memref<2000xf32>
        %1 = affine.load %arg0[%arg12, %arg11] : memref<2000x2000xf32>
        %3 = affine.load %arg9[%arg12] : memref<2000xf32>
        %2 = mulf %arg2, %1 : f32
        %4 = mulf %2, %3 : f32
        %5 = addf %0, %4 : f32
        affine.store %5, %arg8[%arg11] : memref<2000xf32>
      }
    }
    // CHECK: %1 = llvm.mlir.constant(0 : i32) : !llvm.i32
    // CHECK: @matvec_2000x2000x2000(%{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %cst, %1) 
    affine.for %arg11 = 0 to 2000 {
      affine.for %arg12 = 0 to 2000 {
        %0 = affine.load %arg7[%arg11] : memref<2000xf32>
        %1 = affine.load %arg0[%arg11, %arg12] : memref<2000x2000xf32>
        %3 = affine.load %arg8[%arg12] : memref<2000xf32>
        %2 = mulf %arg1, %1 : f32
        %4 = mulf %2, %3 : f32
        %5 = addf %0, %4 : f32
        affine.store %5, %arg7[%arg11] : memref<2000xf32>
      }
    }
    return
}

func @distributed2mm(%arg0: memref<800x1100xf32>,
                     %arg1: memref<1100x900xf32>,
                     %arg2: memref<900x1200xf32>,
                     %arg3: memref<800x1200xf32>,
                     %arg4: f32, %arg5: f32,
                     %arg6: memref<800x900xf32>) {
    // CHECK: %cst_0 = constant 1.000000e+00 : f32
    // D[i][j] *= beta
    affine.for %arg7 = 0 to 800 {
      affine.for %arg8 = 0 to 1200 {
        %0 = affine.load %arg3[%arg7, %arg8] : memref<800x1200xf32>
        %1 = mulf %arg5, %0 : f32
        affine.store %1, %arg3[%arg7, %arg8] : memref<800x1200xf32>
      }
    }
    // tmp[i][j] = 0
    affine.for %arg7 = 0 to 800 {
      affine.for %arg8 = 0 to 900 {
        %cst = constant 0.000000e+00 : f32
        affine.store %cst, %arg6[%arg7, %arg8] : memref<800x900xf32>
      }
    }
    // tmp[i][j] += alpha * A[i][k] * B[k][j]
    // CHECK: %3 = llvm.mlir.constant(0 : i32) : !llvm.i32
    // CHECK: %4 = llvm.mlir.constant(0 : i32) : !llvm.i32
    // CHECK: @matmul_800x900x1100(%3, %4, %{{.*}}, %{{.*}}, %{{.*}}, %arg4, %cst_0, %{{.*}}, %{{.*}}, %{{.*}})
    affine.for %arg7 = 0 to 800 {
      affine.for %arg8 = 0 to 900 {
        affine.for %arg9 = 0 to 1100 {
          %0 = affine.load %arg0[%arg7, %arg9] : memref<800x1100xf32>
          %1 = affine.load %arg1[%arg9, %arg8] : memref<1100x900xf32>
          %2 = affine.load %arg6[%arg7, %arg8] : memref<800x900xf32>
          %3 = mulf %arg4, %0 : f32
          %4 = mulf %3, %1 : f32
          %5 = addf %2, %4 : f32
          affine.store %5, %arg6[%arg7, %arg8] : memref<800x900xf32>
        }
      }
    }
    // D[i][j] += tmp[i][k] * C[k][j]
    // CHECK: %8 = llvm.mlir.constant(0 : i32) : !llvm.i32
    // CHECK: %9 = llvm.mlir.constant(0 : i32) : !llvm.i32
    // CHECK: @matmul_800x1200x900(%8, %9, %{{.*}}, %{{.*}}, %{{.*}}, %cst_0, %cst_0, %{{.*}}, %{{.*}}, %{{.*}})
    affine.for %arg7 = 0 to 800 {
      affine.for %arg8 = 0 to 1200 {
        affine.for %arg9 = 0 to 900 {
          %0 = affine.load %arg6[%arg7, %arg9] : memref<800x900xf32>
          %1 = affine.load %arg2[%arg9, %arg8] : memref<900x1200xf32>
          %2 = affine.load %arg3[%arg7, %arg8] : memref<800x1200xf32>
          %3 = mulf %0, %1 : f32
          %4 = addf %2, %3 : f32
          affine.store %4, %arg3[%arg7, %arg8] : memref<800x1200xf32>
        }
      }
    }
    return
}

func @distributed_gessumv(%arg0: memref<1300x1300xf32>, 
                          %arg1: memref<1300x1300xf32>, 
                          %arg2: f32, %arg3: f32, 
                          %arg4: memref<1300xf32>, %arg5: memref<1300xf32>, 
                          %arg6: memref<1300xf32>) {
  // tmp[i] = 0;
  affine.for %arg7 = 0 to 1300 {
    %cst = constant 0.000000e+00 : f32
    affine.store %cst, %arg6[%arg7] : memref<1300xf32>
  }
  // y[i] = 0;
  affine.for %arg7 = 0 to 1300 {
    %cst = constant 0.000000e+00 : f32
    affine.store %cst, %arg4[%arg7] : memref<1300xf32>
  }
  // y[i] = B[i][j] * x[j] + y[i]
  // CHECK: %0 = llvm.mlir.constant(0 : i32) : !llvm.i32
  // CHECK: @matvec_1300x1300x1300(%arg4, %arg0, %arg5, %cst_0, %cst_0, %0)
  affine.for %arg7 = 0 to 1300 {
    affine.for %arg8 = 0 to 1300 {
      %0 = affine.load %arg0[%arg7, %arg8] : memref<1300x1300xf32>
      %1 = affine.load %arg5[%arg8] : memref<1300xf32>
      %2 = mulf %0, %1 : f32
      %3 = affine.load %arg4[%arg7] : memref<1300xf32>
      %4 = addf %3, %2 : f32
      affine.store %4, %arg4[%arg7] : memref<1300xf32>
    }
  }
  // tmp[i] = A[i][j] * x[j] + tmp[i]
  // CHECK: llvm.mlir.constant(0 : i32) : !llvm.i32
  // CHECK: @matvec_1300x1300x1300(%arg6, %arg1, %arg5, %cst_0, %cst_0, %1)
  affine.for %arg7 = 0 to 1300 {
    affine.for %arg8 = 0 to 1300 {
      %0 = affine.load %arg1[%arg7, %arg8] : memref<1300x1300xf32>
      %1 = affine.load %arg5[%arg8] : memref<1300xf32>
      %2 = mulf %0, %1 : f32
      %3 = affine.load %arg6[%arg7] : memref<1300xf32>
      %4 = addf %3, %2 : f32
      affine.store %4, %arg6[%arg7] : memref<1300xf32>
    }
  }
  // y[i] = alpha * tmp[i] + beta * y[i]
  affine.for %arg7 = 0 to 1300 {
    %0 = affine.load %arg4[%arg7] : memref<1300xf32>
    %1 = mulf %arg2, %0 : f32
    %2 = affine.load %arg6[%arg7] : memref<1300xf32>
    %3 = mulf %arg3, %2 : f32
    %4 = addf %1, %3 : f32
    affine.store %4, %arg6[%arg7] : memref<1300xf32>
  }
  return
}

func @gemver(%arg0: memref<1024x1024xf32>, 
             %arg1: f32, %arg2: f32, 
             %arg3: memref<1024xf32>, %arg4: memref<1024xf32>, 
             %arg5: memref<1024xf32>, %arg6: memref<1024xf32>, 
             %arg7: memref<1024xf32>, %arg8: memref<1024xf32>, 
             %arg9: memref<1024xf32>, %arg10: memref<1024xf32>) {
  // A[i][j] = A[i][j] + u1[i] * V1[j] + u2[i] * v2[j]
  affine.for %arg11 = 0 to 1024 {
    affine.for %arg12 = 0 to 1024 {
      %0 = affine.load %arg0[%arg11, %arg12] : memref<1024x1024xf32>
      %1 = affine.load %arg3[%arg11] : memref<1024xf32>
      %2 = affine.load %arg5[%arg12] : memref<1024xf32>
      %3 = mulf %1, %2 : f32
      %4 = addf %0, %3 : f32
      %5 = affine.load %arg4[%arg11] : memref<1024xf32>
      %6 = affine.load %arg6[%arg12] : memref<1024xf32>
      %7 = mulf %5, %6 : f32
      %8 = addf %4, %7 : f32
      affine.store %8, %arg0[%arg11, %arg12] : memref<1024x1024xf32>
    }
  }
  // x[i] = x[i] + beta * A[j][i] * y[j]
  // CHECK: %0 = llvm.mlir.constant(1 : i32) : !llvm.i32
  // CHECK: @matvec_1024x1024x1024(%arg8, %arg0, %arg9, %arg2, %cst, %0)
  affine.for %arg11 = 0 to 1024 {
    affine.for %arg12 = 0 to 1024 {
      %0 = affine.load %arg8[%arg11] : memref<1024xf32>
      %1 = affine.load %arg0[%arg12, %arg11] : memref<1024x1024xf32>
      %2 = mulf %arg2, %1 : f32
      %3 = affine.load %arg9[%arg12] : memref<1024xf32>
      %4 = mulf %2, %3 : f32
      %5 = addf %0, %4 : f32
      affine.store %5, %arg8[%arg11] : memref<1024xf32>
    }
  }
  // x[i] = x[i] + z[i]
  affine.for %arg11 = 0 to 1024 {
    %0 = affine.load %arg8[%arg11] : memref<1024xf32>
    %1 = affine.load %arg10[%arg11] : memref<1024xf32>
    %2 = addf %0, %1 : f32
    affine.store %2, %arg8[%arg11] : memref<1024xf32>
  }
  // w[i] = w[i] + alpha * A[i][j] * x[j]
  // CHECK: %1 = llvm.mlir.constant(0 : i32) : !llvm.i32
  // CHECK: @matvec_1024x1024x1024(%arg7, %arg0, %arg8, %arg1, %cst, %1)
  affine.for %arg11 = 0 to 1024 {
    affine.for %arg12 = 0 to 1024 {
      %0 = affine.load %arg7[%arg11] : memref<1024xf32>
      %1 = affine.load %arg0[%arg11, %arg12] : memref<1024x1024xf32>
      %2 = mulf %arg1, %1 : f32
      %3 = affine.load %arg8[%arg12] : memref<1024xf32>
      %4 = mulf %2, %3 : f32
      %5 = addf %0, %4 : f32
      affine.store %5, %arg7[%arg11] : memref<1024xf32>
    }
  }
  return
}

func @atax(%arg0: memref<1900x2100xf32>, 
           %arg1: memref<1900xf32>, %arg2: memref<2100xf32>, 
           %arg3: memref<2100xf32>) {
  // CHECK: %cst_0 = constant 1.000000e+00 : f32
  // tmp[i] = 0
  affine.for %arg4 = 0 to 1900 {
    %cst = constant 0.000000e+00 : f32
    affine.store %cst, %arg1[%arg4] : memref<1900xf32>
  }
  // y[i] = 0  
  affine.for %arg4 = 0 to 2100 {
    %cst = constant 0.000000e+00 : f32
    affine.store %cst, %arg3[%arg4] : memref<2100xf32>
  }
  // tmp[i] = tmp[i] + A[i][j] * x[j]
  // CHECK: %0 = llvm.mlir.constant(0 : i32) : !llvm.i32
  // CHECK: @matvec_1900x2100x2100(%arg1, %arg0, %arg2, %cst_0, %cst_0, %0)
  affine.for %arg4 = 0 to 1900 {
    affine.for %arg5 = 0 to 2100 {
      %0 = affine.load %arg1[%arg4] : memref<1900xf32>
      %1 = affine.load %arg0[%arg4, %arg5] : memref<1900x2100xf32>
      %2 = affine.load %arg2[%arg5] : memref<2100xf32>
      %3 = mulf %1, %2 : f32
      %4 = addf %0, %3 : f32
      affine.store %4, %arg1[%arg4] : memref<1900xf32>
    }
  }
  // y[j] = y[j] + A[i][j] * tmp[i]
  // CHECK: %1 = llvm.mlir.constant(1 : i32) : !llvm.i32
  // CHECK: @matvec_1900x2100x1900(%arg3, %arg0, %arg1, %cst_0, %cst_0, %1)
  affine.for %arg4 = 0 to 1900 {
    affine.for %arg5 = 0 to 2100 {
      %0 = affine.load %arg3[%arg5] : memref<2100xf32>
      %1 = affine.load %arg0[%arg4, %arg5] : memref<1900x2100xf32>
      %2 = affine.load %arg1[%arg4] : memref<1900xf32>
      %3 = mulf %1, %2 : f32
      %4 = addf %0, %3 : f32
      affine.store %4, %arg3[%arg5] : memref<2100xf32>
    }
  }
  return
}

func @bicg(%arg0: memref<2100x1900xf32>, 
           %arg1: memref<1900xf32>, %arg2: memref<2100xf32>, 
           %arg3: memref<2100xf32>, %arg4: memref<1900xf32>) {
  // CHECK: %cst_0 = constant 1.000000e+00 : f32
  // q[i] = 0
  affine.for %arg5 = 0 to 2100 {
    %cst = constant 0.000000e+00 : f32
    affine.store %cst, %arg2[%arg5] : memref<2100xf32>
  }
  // s[i] = 0
  affine.for %arg5 = 0 to 1900 {
    %cst = constant 0.000000e+00 : f32
    affine.store %cst, %arg4[%arg5] : memref<1900xf32>
  }
  // s[j] = s[j] + r[i] * A[i][j]
  // Here we don't detect as the tactis for gemv
  // is -> x(i) += A(i, j)T * y(j).
  // Adding the following tactic will solve
  // the issue: s(i) += r(j) * A(i, j)
  affine.for %arg5 = 0 to 2100 {
    affine.for %arg6 = 0 to 1900 {
      %0 = affine.load %arg4[%arg6] : memref<1900xf32>
      %1 = affine.load %arg3[%arg5] : memref<2100xf32>
      %2 = affine.load %arg0[%arg5, %arg6] : memref<2100x1900xf32>
      %3 = mulf %1, %2 : f32
      %4 = addf %0, %3 : f32
      affine.store %4, %arg4[%arg6] : memref<1900xf32>
    }
  }
  // q[i] = q[i] + A[i][j] * p[j]
  // CHECK: %0 = llvm.mlir.constant(0 : i32) : !llvm.i32
  // CHECK: @matvec_2100x1900x1900(%arg2, %arg0, %arg1, %cst_0, %cst_0, %0)
  affine.for %arg5 = 0 to 2100 {
    affine.for %arg6 = 0 to 1900 {
      %0 = affine.load %arg2[%arg5] : memref<2100xf32>
      %1 = affine.load %arg0[%arg5, %arg6] : memref<2100x1900xf32>
      %2 = affine.load %arg1[%arg6] : memref<1900xf32>
      %3 = mulf %1, %2 : f32
      %4 = addf %0, %3 : f32
      affine.store %4, %arg2[%arg5] : memref<2100xf32>
    }
  }
  return
}
