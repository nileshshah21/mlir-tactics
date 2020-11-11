func @contraction.ab.ac.cd(%A: memref<1000x1200xf32>, %B: memref<1200x1100xf32>, %C: memref<1000x1100xf32>) {
  affine.for %i = 0 to 1000 {
    affine.for %j = 0 to 1100 {
      affine.for %k = 0 to 1200 {
        %0 = affine.load %A[%i, %k] : memref<1000x1200xf32>
        %1 = affine.load %B[%k, %j] : memref<1200x1100xf32>
        %2 = affine.load %C[%i, %j] : memref<1000x1100xf32>
        %3 = mulf %0, %1 : f32
        %4 = addf %2, %3 : f32
        affine.store %4, %C[%i, %j] : memref<1000x1100xf32>
      }
    }
  }
  return
}

func @main() {
  %A = alloc() : memref<1000x1200xf32>
  %B = alloc() : memref<1200x1100xf32>
  %C = alloc() : memref<1000x1100xf32>
  
  %cf0 = constant 1.00000e+00 : f32
  %cf1 = constant 2.00000e+00 : f32

  linalg.fill(%A, %cf1) : memref<1000x1200xf32>, f32
  linalg.fill(%B, %cf1) : memref<1200x1100xf32>, f32
  linalg.fill(%C, %cf0) : memref<1000x1100xf32>, f32
  %t_start = call @rtclock() : () -> f64
  call @contraction.ab.ac.cd(%A, %B, %C) : 
    (memref<1000x1200xf32>, memref<1200x1100xf32>, memref<1000x1100xf32>) -> ()
  %t_end = call @rtclock() : () -> f64
  %t = subf %t_end, %t_start : f64
  %num_flops = constant 2640000000 : index
  %num_flops_i = index_cast %num_flops : index to i64
  %num_flops_f = sitofp %num_flops_i : i64 to f64
  %flops = divf %num_flops_f, %t : f64
  call @print_flops(%flops) : (f64) -> ()
  %pC = memref_cast %C : memref<1000x1100xf32> to memref<*xf32>
  //call @print_memref_f32(%pC) : (memref<*xf32>) -> ()
  return 
}

func @print_flops(f64)
func @rtclock() -> f64
