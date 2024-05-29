func @scop_entry(%arg0: memref<2000x2000xf32>,
                 %arg1: memref<2000xf32>,
                 %arg2: memref<2000xf32>,
                 %arg3: memref<2000xf32>,
                 %arg4: memref<2000xf32>) {
    affine.for %arg5 = 0 to 2000 {
      affine.for %arg6 = 0 to 2000 {
        %0 = affine.load %arg1[%arg5] : memref<2000xf32>
        %1 = affine.load %arg0[%arg5, %arg6] : memref<2000x2000xf32>
        %2 = affine.load %arg3[%arg6] : memref<2000xf32>
        %3 = mulf %1, %2 : f32
        %4 = addf %0, %3 : f32
        affine.store %4, %arg1[%arg5] : memref<2000xf32>
      }
    }
    affine.for %arg5 = 0 to 2000 {
      affine.for %arg6 = 0 to 2000 {
        %0 = affine.load %arg2[%arg5] : memref<2000xf32>
        %1 = affine.load %arg0[%arg6, %arg5] : memref<2000x2000xf32>
        %2 = affine.load %arg4[%arg6] : memref<2000xf32>
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
  linalg.fill(%A, %cf2) : memref<2000x2000xf32>, f32

  %t_start = call @rtclock() : () -> f64
  call @scop_entry(%A, %y1, %y2, %x1, %x2) :
    (memref<2000x2000xf32>, memref<2000xf32>, memref<2000xf32>,
     memref<2000xf32>, memref<2000xf32>) -> ()
  %t_end = call @rtclock() : () -> f64
  %t = subf %t_end, %t_start : f64
  //%num_flops = constant 16000000 : index
  //%num_flops_i = index_cast %num_flops : index to i64
  //%num_flops_f = sitofp %num_flops_i : i64 to f64
  //%flops = divf %num_flops_f, %t : f64
  //call @print_flops(%flops) : (f64) -> ()
  //%py1 = memref_cast %y1 : memref<2000xf32> to memref<*xf32>
  //call @print_memref_f32(%py1) : (memref<*xf32>) -> ()
  call @print_double(%t) : (f64) -> ()
  return
}

func @print_flops(f64)
func @print_double(f64)
func @rtclock() -> f64
