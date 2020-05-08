func @scop_entry(%arg0: memref<2000x2000xf32>,
                          %arg1: memref<2000x2000xf32>,
                          %arg2: f32, %arg3: f32,
                          %arg4: memref<2000xf32>, %arg5: memref<2000xf32>,
                          %arg6: memref<2000xf32>) {
  // tmp[i] = 0;
  affine.for %arg7 = 0 to 2000 {
    %cst = constant 0.000000e+00 : f32
    affine.store %cst, %arg6[%arg7] : memref<2000xf32>
  }
  // y[i] = 0;
  affine.for %arg7 = 0 to 2000 {
    %cst = constant 0.000000e+00 : f32
    affine.store %cst, %arg4[%arg7] : memref<2000xf32>
  }
  // y[i] = B[i][j] * x[j] + y[i]
  affine.for %arg7 = 0 to 2000 {
    affine.for %arg8 = 0 to 2000 {
      %0 = affine.load %arg0[%arg7, %arg8] : memref<2000x2000xf32>
      %1 = affine.load %arg5[%arg8] : memref<2000xf32>
      %2 = mulf %0, %1 : f32
      %3 = affine.load %arg4[%arg7] : memref<2000xf32>
      %4 = addf %3, %2 : f32
      affine.store %4, %arg4[%arg7] : memref<2000xf32>
    }
  }
  // tmp[i] = A[i][j] * x[j] + tmp[i]
  affine.for %arg7 = 0 to 2000 {
    affine.for %arg8 = 0 to 2000 {
      %0 = affine.load %arg1[%arg7, %arg8] : memref<2000x2000xf32>
      %1 = affine.load %arg5[%arg8] : memref<2000xf32>
      %2 = mulf %0, %1 : f32
      %3 = affine.load %arg6[%arg7] : memref<2000xf32>
      %4 = addf %3, %2 : f32
      affine.store %4, %arg6[%arg7] : memref<2000xf32>
    }
  }
  // y[i] = alpha * tmp[i] + beta * y[i]
  affine.for %arg7 = 0 to 2000 {
    %0 = affine.load %arg4[%arg7] : memref<2000xf32>
    %1 = mulf %arg2, %0 : f32
    %2 = affine.load %arg6[%arg7] : memref<2000xf32>
    %3 = mulf %arg3, %2 : f32
    %4 = addf %1, %3 : f32
    affine.store %4, %arg6[%arg7] : memref<2000xf32>
  }
  return
}

func @main() {
  %A = alloc() : memref<2000x2000xf32>
  %B = alloc() : memref<2000x2000xf32>
  %tmp = alloc() : memref<2000xf32>
  %y = alloc() : memref<2000xf32>
  %x = alloc() : memref<2000xf32>

  %cf1 = constant 1.00000e+00 : f32
  %cf2 = constant 2.00000e+00 : f32

  linalg.fill(%A, %cf1) : memref<2000x2000xf32>, f32
  linalg.fill(%B, %cf1) : memref<2000x2000xf32>, f32
  linalg.fill(%tmp, %cf1) : memref<2000xf32>, f32
  linalg.fill(%y, %cf2) : memref<2000xf32>, f32
  linalg.fill(%x, %cf2) : memref<2000xf32>, f32

  call @start_timer() : () -> ()
  call @scop_entry(%A, %B, %cf1, %cf2, %tmp, %x, %y) :
    (memref<2000x2000xf32>, memref<2000x2000xf32>, f32, f32, memref<2000xf32>, memref<2000xf32>,
     memref<2000xf32>) -> ()
  call @stop_timer() : () -> ()
  return
}

func @start_timer()
func @stop_timer()

  
  
