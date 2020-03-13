func @contraction.abcd.aebf.fdec(%C: memref<32x32x32x32xf32>,
                                 %A: memref<32x32x32x32xf32>, %B: memref<32x32x32x32xf32>) {
  affine.for %a = 0 to 32 {
    affine.for %b = 0 to 32 {
      affine.for %c = 0 to 32 {
        affine.for %d = 0 to 32 {
          affine.for %e = 0 to 32 {
            affine.for %f = 0 to 32 {
              %0 = affine.load %A[%a, %e, %b, %f] : memref<32x32x32x32xf32>
              %1 = affine.load %B[%f, %d, %e, %c] : memref<32x32x32x32xf32>
              %2 = affine.load %C[%a, %b, %c, %d] : memref<32x32x32x32xf32>
              %3 = mulf %0, %1 : f32
              %4 = addf %2, %3 : f32
              affine.store %4, %C[%a, %b, %c, %d] : memref<32x32x32x32xf32>
            } 
          } 
        } 
      } 
    } 
  } 
  return 
}

func @main() {
  %C = alloc() : memref<32x32x32x32xf32>
  %A = alloc() : memref<32x32x32x32xf32>
  %B = alloc() : memref<32x32x32x32xf32>
  
  %cf0 = constant 1.00000e+00 : f32
  %cf1 = constant 2.00000e+00 : f32

  linalg.fill(%A, %cf1) : memref<32x32x32x32xf32>, f32
  linalg.fill(%B, %cf1) : memref<32x32x32x32xf32>, f32
  linalg.fill(%C, %cf0) : memref<32x32x32x32xf32>, f32

  call @contraction.abcd.aebf.fdec(%C, %A, %B) : 
    (memref<32x32x32x32xf32>, memref<32x32x32x32xf32>, memref<32x32x32x32xf32>) -> ()
  call @print_memref_4d_f32(%C) : (memref<32x32x32x32xf32>) -> ()
  return 
}

func @print_memref_4d_f32(memref<32x32x32x32xf32>)
