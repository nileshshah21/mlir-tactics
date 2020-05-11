func @contraction.abcd.aebf.dfce(%C: memref<32x32x32x32xf32>, 
                                 %A: memref<32x32x32x32xf32>, %B: memref<32x32x32x32xf32>) {
  affine.for %a = 0 to 32 {
    affine.for %b = 0 to 32 {
      affine.for %c = 0 to 32 {
        affine.for %d = 0 to 32 {
          affine.for %e = 0 to 32 {
            affine.for %f = 0 to 32 {
              %0 = affine.load %A[%a, %e, %b, %f] : memref<32x32x32x32xf32>
              %1 = affine.load %B[%d, %f, %c, %e] : memref<32x32x32x32xf32>
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
  %A = alloc() : memref<32x32x32x32xf32>
  %B = alloc() : memref<32x32x32x32xf32>
  %C = alloc() : memref<32x32x32x32xf32>
  
  %cf0 = constant 1.00000e+00 : f32
  %cf1 = constant 2.00000e+00 : f32

  linalg.fill(%A, %cf1) : memref<32x32x32x32xf32>, f32
  linalg.fill(%B, %cf1) : memref<32x32x32x32xf32>, f32
  linalg.fill(%C, %cf0) : memref<32x32x32x32xf32>, f32
  call @start_timer() : () -> ()
  call @contraction.abcd.aebf.dfce(%A, %B, %C) : 
    (memref<32x32x32x32xf32>, memref<32x32x32x32xf32>, memref<32x32x32x32xf32>) -> ()
  call @stop_timer() : () -> ()
  %pC = memref_cast %C : memref<32x32x32x32xf32> to memref<*xf32>
  //call @print_memref_f32(%pC) : (memref<*xf32>) -> ()
  return 
}

func @start_timer()
func @stop_timer()
func @print_memref_f32(memref<*xf32>)
