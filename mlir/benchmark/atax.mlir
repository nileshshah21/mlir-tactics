func @scop_entry(%arg0: memref<2000x2000xf32>,
           %arg1: memref<2000xf32>, %arg2: memref<2000xf32>,
           %arg3: memref<2000xf32>) {
  // tmp[i] = 0
  affine.for %arg4 = 0 to 2000 {
    %cst = constant 0.000000e+00 : f32
    affine.store %cst, %arg1[%arg4] : memref<2000xf32>
  }
  // y[i] = 0  
  affine.for %arg4 = 0 to 2000 {
    %cst = constant 0.000000e+00 : f32
    affine.store %cst, %arg3[%arg4] : memref<2000xf32>
  }
  // tmp[i] = tmp[i] + A[i][j] * x[j]
  affine.for %arg4 = 0 to 2000 {
    affine.for %arg5 = 0 to 2000 {
      %0 = affine.load %arg1[%arg4] : memref<2000xf32>
      %1 = affine.load %arg0[%arg4, %arg5] : memref<2000x2000xf32>
      %2 = affine.load %arg2[%arg5] : memref<2000xf32>
      %3 = mulf %1, %2 : f32
      %4 = addf %0, %3 : f32
      affine.store %4, %arg1[%arg4] : memref<2000xf32>
    }
  }
  // y[j] = y[j] + A[i][j] * tmp[i]
  affine.for %arg4 = 0 to 2000 {
    affine.for %arg5 = 0 to 2000 {
      %0 = affine.load %arg3[%arg5] : memref<2000xf32>
      %1 = affine.load %arg0[%arg4, %arg5] : memref<2000x2000xf32>
      %2 = affine.load %arg1[%arg4] : memref<2000xf32>
      %3 = mulf %1, %2 : f32
      %4 = addf %0, %3 : f32
      affine.store %4, %arg3[%arg5] : memref<2000xf32>
    }
  }
  return
}

func @main() {
  %A = alloc() : memref<2000x2000xf32>
  %tmp = alloc() : memref<2000xf32>
  %x = alloc() : memref<2000xf32>
  %y = alloc() : memref<2000xf32>

  %cf1 = constant 1.00000e+00 : f32
  %cf2 = constant 2.00000e+00 : f32

  linalg.fill(%A, %cf1) : memref<2000x2000xf32>, f32
  linalg.fill(%tmp, %cf1) : memref<2000xf32>, f32
  linalg.fill(%x, %cf2) : memref<2000xf32>, f32
  linalg.fill(%y, %cf2) : memref<2000xf32>, f32
  
  call @start_timer() : () -> ()
  call @scop_entry(%A, %tmp, %x, %y) :
    (memref<2000x2000xf32>, memref<2000xf32>, memref<2000xf32>, memref<2000xf32>) -> ()
  call @stop_timer() : () -> ()
  %pTmp = memref_cast %tmp : memref<2000xf32> to memref<*xf32>
  //call @print_memref_f32(%pTmp) : (memref<*xf32>) -> ()
  return
}

func @start_timer()
func @stop_timer()
func @print_memref_f32(memref<*xf32>)
