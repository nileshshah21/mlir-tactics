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

  %t_start = call @rtclock() : () -> f64
  affine.for %i = 0 to 1000 {
    affine.for %j = 0 to 900 {
      affine.for %k = 0 to 2000 {
        %0 = affine.load %A[%i, %k] : memref<1000x2000xf32>
        %1 = affine.load %B[%k, %j] : memref<2000x900xf32>
        %2 = affine.load %tmp1[%i, %j] : memref<1000x900xf32>
        %3 = mulf %0, %1 : f32
        %4 = addf %2, %3 : f32
        affine.store %4, %tmp1[%i, %j] : memref<1000x900xf32>
      }
    }
  }

  affine.for %i = 0 to 1000 {
    affine.for %j = 0 to 1500 {
      affine.for %k = 0 to 900 {
        %0 = affine.load %tmp1[%i, %k] : memref<1000x900xf32>
        %1 = affine.load %C[%k, %j] : memref<900x1500xf32>
        %2 = affine.load %tmp2[%i, %j] : memref<1000x1500xf32>
        %3 = mulf %0, %1 : f32
        %4 = addf %2, %3 : f32
        affine.store %4, %tmp2[%i, %j] : memref<1000x1500xf32>
      }
    }
  }

  affine.for %i = 0 to 1000 {
    affine.for %j = 0 to 600 {
      affine.for %k = 0 to 1500 {
        %0 = affine.load %tmp2[%i, %k] : memref<1000x1500xf32>
        %1 = affine.load %D[%k, %j] : memref<1500x600xf32>
        %2 = affine.load %tmp3[%i, %j] : memref<1000x600xf32>
        %3 = mulf %0, %1 : f32
        %4 = addf %2, %3 : f32
        affine.store %4, %tmp3[%i, %j] : memref<1000x600xf32>
      }
    }
  }

  affine.for %i = 0 to 1000 {
    affine.for %j = 0 to 800 {
      affine.for %k = 0 to 600 {
        %0 = affine.load %tmp3[%i, %k] : memref<1000x600xf32>
        %1 = affine.load %E[%k, %j] : memref<600x800xf32>
        %2 = affine.load %tmp4[%i, %j] : memref<1000x800xf32>
        %3 = mulf %0, %1 : f32
        %4 = addf %2, %3 : f32
        affine.store %4, %tmp4[%i, %j] : memref<1000x800xf32>
      }
    }
  }
  %t_end = call @rtclock() : () -> f64
  %t = subf %t_end, %t_start : f64
  call @print_double(%t) : (f64) -> ()
  return
}

func @print_double(f64)
func @rtclock() -> f64
