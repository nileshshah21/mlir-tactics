func @main() {

  %A1 = alloc() : memref<1500x400xf32>
  %A2 = alloc() : memref<400x2000xf32>
  %A3 = alloc() : memref<2000x2200xf32>
  %A4 = alloc() : memref<2200x600xf32>
  %A5 = alloc() : memref<600x1400xf32>
  %A6 = alloc() : memref<1400x1000xf32>

  %tmp1 = alloc() : memref<1500x2000xf32>
  %tmp2 = alloc() : memref<1500x2200xf32>
  %tmp3 = alloc() : memref<1500x600xf32>
  %tmp4 = alloc() : memref<1500x1400xf32>
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


  affine.for %i = 0 to 1500 {
    affine.for %j = 0 to 2000 {
      affine.for %k = 0 to 400 {
        %0 = affine.load %A1[%i, %k] : memref<1500x400xf32>
        %1 = affine.load %A2[%k, %j] : memref<400x2000xf32>
        %2 = affine.load %tmp1[%i, %j] : memref<1500x2000xf32>
        %3 = mulf %0, %1 : f32
        %4 = addf %2, %3 : f32
        affine.store %4, %tmp1[%i, %j] : memref<1500x2000xf32>
      }
    }
  }

  affine.for %i = 0 to 1500 {
    affine.for %j = 0 to 2200 {
      affine.for %k = 0 to 2000 {
        %0 = affine.load %tmp1[%i, %k] : memref<1500x2000xf32>
        %1 = affine.load %A3[%k, %j] : memref<2000x2200xf32>
        %2 = affine.load %tmp2[%i, %j] : memref<1500x2200xf32>
        %3 = mulf %0, %1 : f32
        %4 = addf %2, %3 : f32
        affine.store %4, %tmp2[%i, %j] : memref<1500x2200xf32>
      }
    }
  }

  affine.for %i = 0 to 1500 {
    affine.for %j = 0 to 600 {
      affine.for %k = 0 to 2200 {
        %0 = affine.load %tmp2[%i, %k] : memref<1500x2200xf32>
        %1 = affine.load %A4[%k, %j] : memref<2200x600xf32>
        %2 = affine.load %tmp3[%i, %j] : memref<1500x600xf32>
        %3 = mulf %0, %1 : f32
        %4 = addf %2, %3 : f32
        affine.store %4, %tmp3[%i, %j] : memref<1500x600xf32>
      }
    }
  }

  affine.for %i = 0 to 1500 {
    affine.for %j = 0 to 1400 {
      affine.for %k = 0 to 600 {
        %0 = affine.load %tmp3[%i, %k] : memref<1500x600xf32>
        %1 = affine.load %A5[%k, %j] : memref<600x1400xf32>
        %2 = affine.load %tmp4[%i, %j] : memref<1500x1400xf32>
        %3 = mulf %0, %1 : f32
        %4 = addf %2, %3 : f32
        affine.store %4, %tmp4[%i, %j] : memref<1500x1400xf32>
      }
    }
  }

  affine.for %i = 0 to 1500 {
    affine.for %j = 0 to 1000 {
      affine.for %k = 0 to 1400 {
        %0 = affine.load %tmp4[%i, %k] : memref<1500x1400xf32>  
        %1 = affine.load %A6[%k, %j] : memref<1400x1000xf32>
        %2 = affine.load %tmp5[%i, %j] : memref<1500x1000xf32>
        %3 = mulf %0, %1 : f32
        %4 = addf %2, %3 : f32
        affine.store %4, %tmp5[%i, %j] : memref<1500x1000xf32>
      }
    }
  }

  return
}
