func @main() {
  %A = alloc() : memref<1000x2000xf32>
  %B = alloc() : memref<2000x900xf32>
  %tmp1 = alloc() : memref<1000x900xf32>
  %C = alloc() : memref<900x1500xf32>
  %tmp2 = alloc() : memref<1000x1500xf32>
  %D = alloc() : memref<1500x600xf32>
  %tmp3 = alloc() : memref<1000x600xf32>
  %E = alloc() : memref<600x800xf32>
  %tmp4 = alloc() : memref<1000x800xf32>

  %cst = constant 0.000000e+00 : f32
  %cst_0 = constant 1.000000e+00 : f32

  affine.for %arg0 = 0 to 1000 {
    affine.for %arg1 = 0 to 2000 {
      affine.store %cst, %A[%arg0, %arg1] : memref<1000x2000xf32>
    }
  }
  
  affine.for %arg0 = 0 to 2000 {
    affine.for %arg1 = 0 to 900 {
      affine.store %cst, %B[%arg0, %arg1] : memref<2000x900xf32>
    }
  }

  affine.for %arg0 = 0 to 900 {
    affine.for %arg1 = 0 to 1500 {
      affine.store %cst, %C[%arg0, %arg1] : memref<900x1500xf32>
    }
  }
  
  affine.for %arg0 = 0 to 1500 {
    affine.for %arg1 = 0 to 600 {
      affine.store %cst, %D[%arg0, %arg1] : memref<1500x600xf32>
    }
  }

  affine.for %arg0 = 0 to 600 {
    affine.for %arg1 = 0 to 800 {
      affine.store %cst, %E[%arg0, %arg1] : memref<600x800xf32>
    }
  }

  %0 = alloc() : memref<f32>
  %1 = alloc() : memref<f32>
  linalg.fill(%1, %cst_0) : memref<f32>, f32
  linalg.fill(%0, %cst_0) : memref<f32>, f32

  %t_start = call @rtclock() : () -> f64  
  linalg.matmul(%1, %0, %A, %B, %tmp1) :
    memref<f32>, memref<f32>, memref<1000x2000xf32>, memref<2000x900xf32>, memref<1000x900xf32>
  linalg.matmul(%1, %0, %tmp1, %C, %tmp2) :
    memref<f32>, memref<f32>, memref<1000x900xf32>, memref<900x1500xf32>, memref<1000x1500xf32>
  linalg.matmul(%1, %0, %tmp2, %D, %tmp3) :
    memref<f32>, memref<f32>, memref<1000x1500xf32>, memref<1500x600xf32>, memref<1000x600xf32>
  linalg.matmul(%1, %0, %tmp3, %E, %tmp4) :
    memref<f32>, memref<f32>, memref<1000x600xf32>, memref<600x800xf32>, memref<1000x800xf32>
  %t_end = call @rtclock() : () -> f64
  %t = subf %t_end, %t_start : f64
  call @print_double(%t) : (f64) -> ()
  return 
}

func @print_double(f64)
func @rtclock() -> f64
