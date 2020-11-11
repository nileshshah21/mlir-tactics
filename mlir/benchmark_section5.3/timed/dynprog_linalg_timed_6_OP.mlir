func @main() {
  %A1 = alloc() : memref<1500x400xf32>
  %A2 = alloc() : memref<400x2000xf32>
  %A3 = alloc() : memref<2000x2200xf32>
  %A4 = alloc() : memref<2200x600xf32>
  %A5 = alloc() : memref<600x1400xf32>
  %A6 = alloc() : memref<1400x1000xf32>
  
  %tmp1 = alloc() : memref<400x2200xf32>
  %tmp2 = alloc() : memref<400x600xf32>
  %tmp3 = alloc() : memref<400x1400xf32>
  %tmp4 = alloc() : memref<400x1000xf32>
  %tmp5 = alloc() : memref<1500x1000xf32>


  %cst = constant 0.000000e+00 : f32
  %cst_0 = constant 1.000000e+00 : f32

  affine.for %arg0 = 0 to 1500 {
    affine.for %arg1 = 0 to 400 {
      affine.store %cst, %A1[%arg0, %arg1] : memref<1500x400xf32>
    }
  }
  
  affine.for %arg0 = 0 to 400 {
    affine.for %arg1 = 0 to 2000 {
      affine.store %cst, %A2[%arg0, %arg1] : memref<400x2000xf32>
    }
  }

  affine.for %arg0 = 0 to 2000 {
    affine.for %arg1 = 0 to 2200 {
      affine.store %cst, %A3[%arg0, %arg1] : memref<2000x2200xf32>
    }
  }
  
  affine.for %arg0 = 0 to 2200 {
    affine.for %arg1 = 0 to 600 {
      affine.store %cst, %A4[%arg0, %arg1] : memref<2200x600xf32>
    }
  }

  affine.for %arg0 = 0 to 600 {
    affine.for %arg1 = 0 to 1400 {
      affine.store %cst, %A5[%arg0, %arg1] : memref<600x1400xf32>
    }
  }

  affine.for %arg0 = 0 to 1400 {
    affine.for %arg1 = 0 to 1000 {
      affine.store %cst, %A6[%arg0, %arg1] : memref<1400x1000xf32>
    }
  }

  %0 = alloc() : memref<f32>
  %1 = alloc() : memref<f32>
  linalg.fill(%1, %cst_0) : memref<f32>, f32
  linalg.fill(%0, %cst_0) : memref<f32>, f32

  %t_start = call @rtclock() : () -> f64  

  linalg.matmul(%1, %0, %A2, %A3, %tmp1) :
    memref<f32>, memref<f32>, memref<400x2000xf32>, memref<2000x2200xf32>, memref<400x2200xf32>
  linalg.matmul(%1, %0, %tmp1, %A4, %tmp2) :
    memref<f32>, memref<f32>, memref<400x2200xf32>, memref<2200x600xf32>, memref<400x600xf32>
  linalg.matmul(%1, %0, %tmp2, %A5, %tmp3) :
    memref<f32>, memref<f32>, memref<400x600xf32>, memref<600x1400xf32>, memref<400x1400xf32>
  linalg.matmul(%1, %0, %tmp3, %A6, %tmp4) :
    memref<f32>, memref<f32>, memref<400x1400xf32>, memref<1400x1000xf32>, memref<400x1000xf32>
  linalg.matmul(%1, %0, %A1, %tmp4, %tmp5) :
    memref<f32>, memref<f32>, memref<1500x400xf32>, memref<400x1000xf32>, memref<1500x1000xf32>

  %t_end = call @rtclock() : () -> f64
  %t = subf %t_end, %t_start : f64
  call @print_double(%t) : (f64) -> ()
  return 
}

func @print_double(f64)
func @rtclock() -> f64
