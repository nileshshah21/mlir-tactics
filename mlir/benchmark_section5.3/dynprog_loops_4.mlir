func @chainMatmul() {
    %arg0 = alloc() : memref<800x1100xf32>
    %arg1 = alloc() : memref<1100x900xf32>
    %arg2 = alloc() : memref<900x1200xf32>
    %arg3 = alloc() : memref<1200x100xf32>
    %arg4 = alloc() : memref<800x900xf32>
    %arg5 = alloc() : memref<800x1200xf32>
    %arg6 = alloc() : memref<800x100xf32>

    %cst = constant 1.000000e+00 : f32
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

    affine.for %i = 0 to 800 {
      affine.for %j = 0 to 900 {
        affine.for %k = 0 to 1100 {
	        %0 = affine.load %arg0[%i, %k] : memref<800x1100xf32>
	        %1 = affine.load %arg1[%k, %j] : memref<1100x900xf32>
	        %2 = affine.load %arg4[%i, %j] : memref<800x900xf32>
	        %3 = mulf %0, %1 : f32
	        %4 = addf %2, %3 : f32
	        affine.store %4, %arg4[%i, %j] : memref<800x900xf32>
	      }
      }
    }

  affine.for %i = 0 to 800 {
    affine.for %j = 0 to 1200 {
      affine.for %k = 0 to 900 {
        %0 = affine.load %arg4[%i, %k] : memref<800x900xf32>
        %1 = affine.load %arg2[%k, %j] : memref<900x1200xf32>
        %2 = affine.load %arg5[%i, %j] : memref<800x1200xf32>
        %3 = mulf %0, %1 : f32
        %4 = addf %2, %3 : f32
        affine.store %4, %arg5[%i, %j] : memref<800x1200xf32>
      }
    }
  }

  affine.for %i = 0 to 800 {
    affine.for %j = 0 to 100 {
      affine.for %k = 0 to 1200 {
        %0 = affine.load %arg5[%i, %k] : memref<800x1200xf32>
        %1 = affine.load %arg3[%k, %j] : memref<1200x100xf32>
        %2 = affine.load %arg6[%i, %j] : memref<800x100xf32>
        %3 = mulf %0, %1 : f32
        %4 = addf %2, %3 : f32
        affine.store %4, %arg6[%i, %j] : memref<800x100xf32>
      }
    }
  }

  return
}  
