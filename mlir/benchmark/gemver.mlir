
func @scop_entry(%arg0: memref<2000x2000xf32>, 
                 %arg1: f32, %arg2: f32, 
                 %arg3: memref<2000xf32>, 
                 %arg4: memref<2000xf32>, %arg5: memref<2000xf32>, 
                 %arg6: memref<2000xf32>, %arg7: memref<2000xf32>, 
                 %arg8: memref<2000xf32>, %arg9: memref<2000xf32>, %arg10: memref<2000xf32>) {
    affine.for %arg11 = 0 to 2000 {
      affine.for %arg12 = 0 to 2000 {
        %0 = affine.load %arg0[%arg11, %arg12] : memref<2000x2000xf32>
        %1 = affine.load %arg3[%arg11] : memref<2000xf32>
        %2 = affine.load %arg5[%arg12] : memref<2000xf32>
        %3 = mulf %1, %2 : f32
        %4 = addf %0, %3 : f32
        %5 = affine.load %arg4[%arg11] : memref<2000xf32>
        %6 = affine.load %arg6[%arg12] : memref<2000xf32>
        %7 = mulf %5, %6 : f32
        %8 = addf %4, %7 : f32
        affine.store %8, %arg0[%arg11, %arg12] : memref<2000x2000xf32>
      }
    }
    affine.for %arg11 = 0 to 2000 {
      affine.for %arg12 = 0 to 2000 {
        %0 = affine.load %arg8[%arg11] : memref<2000xf32>
        %1 = affine.load %arg0[%arg12, %arg11] : memref<2000x2000xf32>
        %3 = affine.load %arg9[%arg12] : memref<2000xf32>
        %2 = mulf %arg2, %1 : f32 
        %4 = mulf %2, %3 : f32
        %5 = addf %0, %4 : f32
        affine.store %5, %arg8[%arg11] : memref<2000xf32>
      }
    }
    affine.for %arg11 = 0 to 2000 {
      %0 = affine.load %arg8[%arg11] : memref<2000xf32>
      %1 = affine.load %arg10[%arg11] : memref<2000xf32>
      %2 = addf %0, %1 : f32
      affine.store %2, %arg8[%arg11] : memref<2000xf32>
    }
    affine.for %arg11 = 0 to 2000 {
      affine.for %arg12 = 0 to 2000 {
        %0 = affine.load %arg7[%arg11] : memref<2000xf32>
        %1 = affine.load %arg0[%arg11, %arg12] : memref<2000x2000xf32>
        %3 = affine.load %arg8[%arg12] : memref<2000xf32>
        %2 = mulf %arg1, %1 : f32 
        %4 = mulf %2, %3 : f32
        %5 = addf %0, %4 : f32
        affine.store %5, %arg7[%arg11] : memref<2000xf32>
      }
    }
    return
}

func @main() {
  %A = alloc() : memref<2000x2000xf32>
  %u1 = alloc() : memref<2000xf32>
  %u2 = alloc() : memref<2000xf32>
  %v1 = alloc() : memref<2000xf32>
  %v2 = alloc() : memref<2000xf32>
  %w = alloc() : memref<2000xf32>
  %x = alloc() : memref<2000xf32>
  %y = alloc() : memref<2000xf32>
  %z = alloc() : memref<2000xf32>

  %cf1 = constant 1.00000e+00 : f32
  %cf2 = constant 2.00000e+00 : f32

  linalg.fill(%A, %cf1) : memref<2000x2000xf32>, f32
  linalg.fill(%u1, %cf1) : memref<2000xf32>, f32
  linalg.fill(%u2, %cf2) : memref<2000xf32>, f32
  linalg.fill(%v1, %cf2) : memref<2000xf32>, f32
  linalg.fill(%v2, %cf2) : memref<2000xf32>, f32
  linalg.fill(%w, %cf2) : memref<2000xf32>, f32
  linalg.fill(%x, %cf2) : memref<2000xf32>, f32
  linalg.fill(%y, %cf2) : memref<2000xf32>, f32
  linalg.fill(%z, %cf2) : memref<2000xf32>, f32
  
  call @start_timer() : () -> ()
  call @scop_entry(%A, %cf1, %cf2, %u1, %u2, %v1, %v2, %w, %x, %y, %z) :
    (memref<2000x2000xf32>, f32, f32, memref<2000xf32>, memref<2000xf32>,
     memref<2000xf32>, memref<2000xf32>, memref<2000xf32>, memref<2000xf32>, 
     memref<2000xf32>, memref<2000xf32>) -> ()
  call @stop_timer() : () -> ()
  return
}

func @start_timer()
func @stop_timer()

  
  
