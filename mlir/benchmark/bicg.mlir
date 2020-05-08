func @scop_entry(%arg0: memref<2000x2000xf32>,
           %arg1: memref<2000xf32>, %arg2: memref<2000xf32>,
           %arg3: memref<2000xf32>, %arg4: memref<2000xf32>) {
  // q[i] = 0
  affine.for %arg5 = 0 to 2000 {
    %cst = constant 0.000000e+00 : f32
    affine.store %cst, %arg2[%arg5] : memref<2000xf32>
  }
  // s[i] = 0
  affine.for %arg5 = 0 to 2000 {
    %cst = constant 0.000000e+00 : f32
    affine.store %cst, %arg4[%arg5] : memref<2000xf32>
  }
  // s[j] = s[j] + r[i] * A[i][j]
  // Here we don't detect as the tactis for gemv
  // is -> x(i) += A(i, j)T * y(j).
  // Adding the following tactic will solve
  // the issue: s(i) += r(j) * A(i, j)
  affine.for %arg5 = 0 to 2000 {
    affine.for %arg6 = 0 to 2000 {
      %0 = affine.load %arg4[%arg6] : memref<2000xf32>
      %1 = affine.load %arg3[%arg5] : memref<2000xf32>
      %2 = affine.load %arg0[%arg5, %arg6] : memref<2000x2000xf32>
      %3 = mulf %1, %2 : f32
      %4 = addf %0, %3 : f32
      affine.store %4, %arg4[%arg6] : memref<2000xf32>
    }
  }
  // q[i] = q[i] + A[i][j] * p[j]
  affine.for %arg5 = 0 to 2000 {
    affine.for %arg6 = 0 to 2000 {
      %0 = affine.load %arg2[%arg5] : memref<2000xf32>
      %1 = affine.load %arg0[%arg5, %arg6] : memref<2000x2000xf32>
      %2 = affine.load %arg1[%arg6] : memref<2000xf32>
      %3 = mulf %1, %2 : f32
      %4 = addf %0, %3 : f32
      affine.store %4, %arg2[%arg5] : memref<2000xf32>
    }
  }
  return
}

func @main() {
  %x1 = alloc() : memref<2000xf32>
  %y1 = alloc() : memref<2000xf32>
  %x2 = alloc() : memref<2000xf32>
  %y2 = alloc() : memref<2000xf32>
  %A = alloc() : memref<2000x2000xf32>
  
  %cf1 = constant 1.00000e+00 : f32
  %cf2 = constant 2.00000e+00 : f32
  
  linalg.fill(%x1, %cf1) : memref<2000xf32>, f32
  linalg.fill(%y1, %cf1) : memref<2000xf32>, f32
  linalg.fill(%y2, %cf2) : memref<2000xf32>, f32
  linalg.fill(%x2, %cf2) : memref<2000xf32>, f32
  
  call @start_timer() : () -> ()
  call @scop_entry(%A, %x1, %x1, %y1, %y2) : 
    (memref<2000x2000xf32>, memref<2000xf32>, memref<2000xf32>,
     memref<2000xf32>, memref<2000xf32>) -> ()
  call @stop_timer() : () -> ()
  return
}

func @start_timer()
func @stop_timer() 
