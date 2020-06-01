func @chainMatmul() {
    %arg0 = alloc() : memref<800x1100xf32>
    %arg1 = alloc() : memref<1100x900xf32>
    %arg2 = alloc() : memref<900x1200xf32>
    %arg3 = alloc() : memref<1200x100xf32>
    %arg4 = alloc() : memref<800x900xf32>
    %arg5 = alloc() : memref<800x1200xf32>
    %arg6 = alloc() : memref<800x100xf32>

    %cst = constant 0.000000e+00 : f32
    %cst_0 = constant 1.000000e+00 : f32
    affine.for %arg7 = 0 to 800 {
      affine.for %arg8 = 0 to 1100 {
        affine.store %cst, %arg0[%arg7, %arg8] : memref<800x1100xf32>
      }
    }
    affine.for %arg7 = 0 to 1100 {
      affine.for %arg8 = 0 to 900 {
        affine.store %cst, %arg1[%arg7, %arg8] : memref<1100x900xf32>
      }
    }
    affine.for %arg7 = 0 to 900 {
      affine.for %arg8 = 0 to 1200 {
        affine.store %cst, %arg2[%arg7, %arg8] : memref<900x1200xf32>
      }
    }
    affine.for %arg7 = 0 to 1200 {
      affine.for %arg8 = 0 to 100 {
        affine.store %cst, %arg3[%arg7, %arg8] : memref<1200x100xf32>
      }
    }
    %0 = alloc() : memref<f32>
    %1 = alloc() : memref<f32>
    linalg.fill(%1, %cst_0) : memref<f32>, f32
    linalg.fill(%0, %cst_0) : memref<f32>, f32
    linalg.matmul(%1, %0, %arg0, %arg1, %arg4) :
      memref<f32>, memref<f32>, memref<800x1100xf32>, memref<1100x900xf32>, memref<800x900xf32>
    %2 = alloc() : memref<f32>
    %3 = alloc() : memref<f32>
    linalg.fill(%3, %cst_0) : memref<f32>, f32
    linalg.fill(%2, %cst_0) : memref<f32>, f32
    linalg.matmul(%3, %2, %arg4, %arg2, %arg5) :
      memref<f32>, memref<f32>, memref<800x900xf32>, memref<900x1200xf32>, memref<800x1200xf32>
    %4 = alloc() : memref<f32>
    %5 = alloc() : memref<f32>
    linalg.fill(%5, %cst_0) : memref<f32>, f32
    linalg.fill(%4, %cst_0) : memref<f32>, f32
    linalg.matmul(%5, %4, %arg5, %arg3, %arg6) :
      memref<f32>, memref<f32>, memref<800x1200xf32>, memref<1200x100xf32>, memref<800x100xf32>
    return
}
