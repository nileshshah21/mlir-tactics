func @contraction.ab.ac.cd(%A: memref<32x64xf32>, %B: memref<64x32xf32>, %C: memref<32x32xf32>) {
  affine.for %i = 0 to 32 {
    affine.for %j = 0 to 32 {
      affine.for %k = 0 to 64 {
        %0 = affine.load %A[%i, %k] : memref<32x64xf32>
        %1 = affine.load %B[%k, %j] : memref<64x32xf32>
        %2 = affine.load %C[%i, %j] : memref<32x32xf32>
        %3 = mulf %0, %1 : f32
        %4 = addf %2, %3 : f32
        affine.store %4, %C[%i, %j] : memref<32x32xf32>
      }
    }
  }
  return
}

func @main() {
  %A = alloc() : memref<32x64xf32>
  %B = alloc() : memref<64x32xf32>
  %C = alloc() : memref<32x32xf32>
  
  %cf0 = constant 1.00000e+00 : f32
  %cf1 = constant 2.00000e+00 : f32

  linalg.fill(%A, %cf1) : memref<32x64xf32>, f32
  linalg.fill(%B, %cf1) : memref<64x32xf32>, f32
  linalg.fill(%C, %cf0) : memref<32x32xf32>, f32
  call @contraction.ab.ac.cd(%A, %B, %C) : 
    (memref<32x64xf32>, memref<64x32xf32>, memref<32x32xf32>) -> ()
  call @print_memref_2d_f32(%C) : (memref<32x32xf32>) -> ()
  return 
}

func @print_memref_2d_f32(memref<32x32xf32>)
