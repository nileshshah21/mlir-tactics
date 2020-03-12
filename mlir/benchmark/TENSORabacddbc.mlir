func @contraction.ab.acd.dbc(%C: memref<1024x1024xf32>, %A: memref<1024x32x32xf32>, %B: memref<32x1024x32xf32>) {
  affine.for %a = 0 to 1024 {
    affine.for %b = 0 to 1024 {
      affine.for %c = 0 to 32 {
        affine.for %d = 0 to 32 {
          %0 = affine.load %A[%a, %c, %d] : memref<1024x32x32xf32>
          %1 = affine.load %B[%d, %b, %c] : memref<32x1024x32xf32>
          %2 = affine.load %C[%a, %b] : memref<1024x1024xf32>
          %3 = mulf %0, %1 : f32
          %4 = addf %2, %3 : f32
          affine.store %4, %C[%a, %b] : memref<1024x1024xf32>
        }
      }
    }
  }
  return
}

func @main() {
  %C = alloc() : memref<1024x1024xf32>
  %A = alloc() : memref<1024x32x32xf32>
  %B = alloc() : memref<32x1024x32xf32>
  
  %cf0 = constant 1.00000e+00 : f32
  %cf1 = constant 2.00000e+00 : f32

  linalg.fill(%A, %cf1) : memref<1024x32x32xf32>, f32
  linalg.fill(%B, %cf1) : memref<32x1024x32xf32>, f32
  linalg.fill(%C, %cf0) : memref<1024x1024xf32>, f32

  call @contraction.ab.acd.dbc(%C, %A, %B) : (memref<1024x1024xf32>, memref<1024x32x32xf32>, memref<32x1024x32xf32>) -> ()
  call @print_memref_2d_f32(%C) : (memref<1024x1024xf32>) -> ()
  return 
}

func @print_memref_2d_f32(memref<1024x1024xf32>)
