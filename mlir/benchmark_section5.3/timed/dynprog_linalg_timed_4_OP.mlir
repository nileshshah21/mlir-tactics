func @main() {
    %cst = constant 1.000000e+00 : f32
    %0 = alloc() : memref<800x1100xf32>
    %1 = alloc() : memref<1100x900xf32>
    %2 = alloc() : memref<900x1200xf32>
    %3 = alloc() : memref<1200x100xf32>
    affine.for %arg0 = 0 to 800 {
      affine.for %arg1 = 0 to 1100 {
        affine.store %cst, %0[%arg0, %arg1] : memref<800x1100xf32>
      }
    }
    affine.for %arg0 = 0 to 1100 {
      affine.for %arg1 = 0 to 900 {
        affine.store %cst, %1[%arg0, %arg1] : memref<1100x900xf32>
      }
    }
    affine.for %arg0 = 0 to 900 {
      affine.for %arg1 = 0 to 1200 {
        affine.store %cst, %2[%arg0, %arg1] : memref<900x1200xf32>
      }
    }
    affine.for %arg0 = 0 to 1200 {
      affine.for %arg1 = 0 to 100 {
        affine.store %cst, %3[%arg0, %arg1] : memref<1200x100xf32>
      }
    }
    %4 = alloc() : memref<f32>
    %5 = alloc() : memref<f32>
    linalg.fill(%5, %cst) : memref<f32>, f32
    linalg.fill(%4, %cst) : memref<f32>, f32
    %6 = alloc() : memref<f32>
    %7 = alloc() : memref<f32>
    linalg.fill(%7, %cst) : memref<f32>, f32
    linalg.fill(%6, %cst) : memref<f32>, f32
    %8 = alloc() : memref<f32>
    %9 = alloc() : memref<f32>
    linalg.fill(%9, %cst) : memref<f32>, f32
    linalg.fill(%8, %cst) : memref<f32>, f32
    %10 = alloc() : memref<900x100xf32>
    %11 = alloc() : memref<1100x100xf32>
    %12 = alloc() : memref<800x100xf32>
    %t_start = call @rtclock() : () -> f64
    linalg.matmul(%4, %5, %2, %3, %10) : memref<f32>, memref<f32>, memref<900x1200xf32>, memref<1200x100xf32>, memref<900x100xf32>
    linalg.matmul(%6, %7, %1, %10, %11) : memref<f32>, memref<f32>, memref<1100x900xf32>, memref<900x100xf32>, memref<1100x100xf32>
    linalg.matmul(%8, %9, %0, %11, %12) : memref<f32>, memref<f32>, memref<800x1100xf32>, memref<1100x100xf32>, memref<800x100xf32>
    %t_end = call @rtclock() : () -> f64
    %t = subf %t_end, %t_start : f64
    call @print_double(%t) : (f64) -> ()
    return
}

func @print_double(f64)
func @rtclock() -> f64
