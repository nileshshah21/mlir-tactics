
func @scop_entry(%arg0: memref<800x1100xf32>, 
                 %arg1: memref<1100x900xf32>, 
                 %arg2: memref<900x1200xf32>, 
                 %arg3: memref<800x1200xf32>, 
                 %arg4: f32, %arg5: f32, 
                 %arg6: memref<800x900xf32>) {
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
    affine.for %arg7 = 0 to 800 {
      affine.for %arg8 = 0 to 1200 {
        affine.for %arg9 = 0 to 900 {
          %0 = affine.load %arg6[%arg7, %arg9] : memref<800x900xf32>
          %1 = affine.load %arg2[%arg9, %arg8] : memref<900x1200xf32>
          %3 = affine.load %arg3[%arg7, %arg8] : memref<800x1200xf32>
          %2 = mulf %0, %1 : f32 
          %4 = addf %3, %2 : f32
          affine.store %4, %arg3[%arg7, %arg8] : memref<800x1200xf32>
        }
      }
    }
    return
}

func @main() {
  %tmp = alloc() : memref<800x900xf32>
  %A = alloc() : memref<800x1100xf32>
  %B = alloc() : memref<1100x900xf32>
  %D = alloc() : memref<800x1200xf32>
  %C = alloc() : memref<900x1200xf32>
  
  %cf1 = constant 1.00000e+00 : f32
  %cf2 = constant 2.00000e+00 : f32 

  linalg.fill(%tmp, %cf1) : memref<800x900xf32>, f32
  linalg.fill(%A, %cf1) : memref<800x1100xf32>, f32
  linalg.fill(%B, %cf2) : memref<1100x900xf32>, f32
  linalg.fill(%D, %cf2) : memref<800x1200xf32>, f32
  linalg.fill(%C, %cf2) : memref<900x1200xf32>, f32
  
  call @start_timer() : () -> ()
  call @scop_entry(%A, %B, %C, %D, %cf1, %cf2, %tmp) :
    (memref<800x1100xf32>, memref<1100x900xf32>, 
     memref<900x1200xf32>, memref<800x1200xf32>, f32, f32, memref<800x900xf32>) -> ()
  call @stop_timer() : () -> ()
  return 
}

func @start_timer()
func @stop_timer() 

