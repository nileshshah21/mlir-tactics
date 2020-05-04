func @scop_entry(%arg0: memref<800x1000xf32>, 
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
    affine.for %arg7 = 0 to 800 {
      affine.for %arg8 = 0 to 900 {
        affine.for %arg9 = 0 to 1000 {
          %0 = affine.load %arg0[%arg7, %arg9] : memref<800x1000xf32>
          %1 = affine.load %arg1[%arg9, %arg8] : memref<1000x900xf32>
          %3 = affine.load %arg4[%arg7, %arg8] : memref<800x900xf32>
          %2 = mulf %0, %1 : f32 
          %4 = addf %2, %3 : f32
          affine.store %4, %arg4[%arg7, %arg8] : memref<800x900xf32>
        }
      }
    }
    // F[i][j] += C[i][k] * D[k][j]
    affine.for %arg7 = 0 to 900 {
      affine.for %arg8 = 0 to 1100 {
        affine.for %arg9 = 0 to 1200 {
          %0 = affine.load %arg2[%arg7, %arg9] : memref<900x1200xf32>
          %1 = affine.load %arg3[%arg9, %arg8] : memref<1200x1100xf32>
          %3 = affine.load %arg5[%arg7, %arg8] : memref<900x1100xf32>
          %2 = mulf %0, %1 : f32
          %4 = addf %2, %3 : f32
          affine.store %4, %arg5[%arg7, %arg8] : memref<900x1100xf32>
        }
      }
    }
    // G[i][j] += E[i][k] * F[k][j]
    affine.for %arg7 = 0 to 800 {
      affine.for %arg8 = 0 to 1100 {
        affine.for %arg9 = 0 to 900 {
          %0 = affine.load %arg4[%arg7, %arg9] : memref<800x900xf32>
          %1 = affine.load %arg5[%arg9, %arg8] : memref<900x1100xf32>
          %3 = affine.load %arg6[%arg7, %arg8] : memref<800x1100xf32>
          %2 = mulf %0, %1 : f32 
          %4 = addf %2, %3 : f32
          affine.store %4, %arg6[%arg7, %arg8] : memref<800x1100xf32>
        }
      }
    }
    return
}

func @main() {
  %E = alloc() : memref<800x900xf32>
  %A = alloc() : memref<800x1000xf32>
  %B = alloc() : memref<1000x900xf32>
  %F = alloc() : memref<900x1100xf32>
  %C = alloc() : memref<900x1200xf32>
  %D = alloc() : memref<1200x1100xf32>
  %G = alloc() : memref<800x1100xf32>
  
  %cf1 = constant 1.00000e+00 : f32
  %cf2 = constant 2.00000e+00 : f32

  linalg.fill(%E, %cf1) : memref<800x900xf32>, f32
  linalg.fill(%A, %cf2) : memref<800x1000xf32>, f32
  linalg.fill(%B, %cf2) : memref<1000x900xf32>, f32
  linalg.fill(%F, %cf2) : memref<900x1100xf32>, f32
  linalg.fill(%C, %cf1) : memref<900x1200xf32>, f32
  linalg.fill(%D, %cf1) : memref<1200x1100xf32>, f32
  linalg.fill(%G, %cf2) : memref<800x1100xf32>, f32

  call @start_timer() : () -> ()
  call @scop_entry(%A, %B, %C, %D, %E, %F, %G) :
    (memref<800x1000xf32>, memref<1000x900xf32>, memref<900x1200xf32>,
     memref<1200x1100xf32>, memref<800x900xf32>, memref<900x1100xf32>, memref<800x1100xf32>) -> ()
  call @stop_timer() : () -> ()
  return
}

func @start_timer()
func @stop_timer()



