func @scop_entry(%A: memref<2100x1900xf32>,
           %p: memref<1900xf32>, %q: memref<2100xf32>,
           %r: memref<2100xf32>, %s: memref<1900xf32>) {
  // q[i] = 0
  affine.for %arg5 = 0 to 2100 {
    %cst = constant 0.000000e+00 : f32
    affine.store %cst, %q[%arg5] : memref<2100xf32>
  }
  // s[i] = 0
  affine.for %arg5 = 0 to 1900 {
    %cst = constant 0.000000e+00 : f32
    affine.store %cst, %s[%arg5] : memref<1900xf32>
  }
  // s[j] = s[j] + r[i] * A[i][j]
  // Here we don't detect as the tactis for gemv
  // is -> x(i) += A(i, j)T * y(j).
  // Adding the following tactic will solve
  // the issue: s(i) += r(j) * A(i, j)
  affine.for %arg5 = 0 to 2100 {
    affine.for %arg6 = 0 to 1900 {
      %0 = affine.load %s[%arg6] : memref<1900xf32>
      %1 = affine.load %r[%arg5] : memref<2100xf32>
      %2 = affine.load %A[%arg5, %arg6] : memref<2100x1900xf32>
      %3 = mulf %1, %2 : f32
      %4 = addf %0, %3 : f32
      affine.store %4, %s[%arg6] : memref<1900xf32>
    }
  }
  // q[i] = q[i] + A[i][j] * p[j]
  affine.for %arg5 = 0 to 2100 {
    affine.for %arg6 = 0 to 1900 {
      %0 = affine.load %q[%arg5] : memref<2100xf32>
      %1 = affine.load %A[%arg5, %arg6] : memref<2100x1900xf32>
      %2 = affine.load %p[%arg6] : memref<1900xf32>
      %3 = mulf %1, %2 : f32
      %4 = addf %0, %3 : f32
      affine.store %4, %q[%arg5] : memref<2100xf32>
    }
  }
  return
}

func @main() {
  %p = alloc() : memref<1900xf32>
  %r = alloc() : memref<2100xf32>
  %q = alloc() : memref<2100xf32>
  %s = alloc() : memref<1900xf32>
  %A = alloc() : memref<2100x1900xf32>
  
  %cf1 = constant 1.00000e+00 : f32
  %cf2 = constant 2.00000e+00 : f32
  
  linalg.fill(%p, %cf1) : memref<1900xf32>, f32
  linalg.fill(%r, %cf1) : memref<2100xf32>, f32
  linalg.fill(%q, %cf2) : memref<2100xf32>, f32
  linalg.fill(%s, %cf2) : memref<1900xf32>, f32
  linalg.fill(%A, %cf1) : memref<2100x1900xf32>, f32
  
  call @start_timer() : () -> ()
  call @scop_entry(%A, %p, %q, %r, %s) : 
    (memref<2100x1900xf32>, memref<1900xf32>, memref<2100xf32>,
     memref<2100xf32>, memref<1900xf32>) -> ()
  %qC = memref_cast %q : memref<2100xf32> to memref<*xf32>
  //call @print_memref_f32(%qC) : (memref<*xf32>) -> ()
  call @stop_timer() : () -> ()
  return
}

func @start_timer()
func @stop_timer() 
func @print_memref_f32(memref<*xf32>)
