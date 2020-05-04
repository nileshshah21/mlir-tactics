func @jacobi1d(%A: memref<1024xf32>, %B: memref<1024xf32>) {
  %cf = constant 0.333333e+00 : f32 
  affine.for %t = 0 to 10 {
    affine.for %i = 1 to 1023 {
      %1 = affine.load %A[%i + 1] : memref<1024xf32>
      %2 = affine.load %A[%i] : memref<1024xf32>
      %3 = affine.load %A[%i - 1] : memref<1024xf32>
      %4 = addf %1, %2 : f32
      %5 = addf %4, %3 : f32
      %6 = mulf %5, %cf : f32
      affine.store %4, %B[%i] : memref<1024xf32>
    }
  }
  return 
}
      
func @main() {
  %A = alloc() : memref<1024xf32>
  %B = alloc() : memref<1024xf32>
  
  %cf0 = constant 1.00000e+00 : f32
  %cf1 = constant 2.00000e+00 : f32

  linalg.fill(%A, %cf1) : memref<1024xf32>, f32
  linalg.fill(%B, %cf0) : memref<1024xf32>, f32

  call @jacobi1d(%A, %B) : (memref<1024xf32>, memref<1024xf32>) -> ()
  return 
}
