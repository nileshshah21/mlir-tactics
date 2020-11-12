func @contraction.ab.ac.cd(%A: memref<1200000xf32>, %B: memref<1320000xf32>, %C: memref<1100000xf32>) {
  %K = constant 1200 : index
  %J = constant 1100 : index
  affine.for %i = 0 to 1000 {
    affine.for %j = 0 to 1100 {
      affine.for %k = 0 to 1200 {
        %indexA = muli %i, %K : index
        %indexA_f = addi %indexA, %k : index
        %indexB = muli %k, %J : index
        %indexB_f = addi %indexB, %j : index
        %indexC = muli %i, %J : index
        %indexC_f = addi %indexC, %j : index
        %0 = load %A[%indexA_f] : memref<1200000xf32>
        %1 = load %B[%indexB_f] : memref<1320000xf32>
        %2 = load %C[%indexC_f] : memref<1100000xf32>
        %3 = mulf %0, %1 : f32
        %4 = addf %2, %3 : f32
        store %4, %C[%indexC_f] : memref<1100000xf32>
      }
    }
  }
  return
}

func @main() {
  %A = alloc() : memref<1200000xf32>
  %B = alloc() : memref<1320000xf32>
  %C = alloc() : memref<1100000xf32>
  
  %cf0 = constant 1.00000e+00 : f32
  %cf1 = constant 2.00000e+00 : f32

  linalg.fill(%A, %cf1) : memref<1200000xf32>, f32
  linalg.fill(%B, %cf1) : memref<1320000xf32>, f32
  linalg.fill(%C, %cf0) : memref<1100000xf32>, f32
  %t_start = call @rtclock() : () -> f64
  call @contraction.ab.ac.cd(%A, %B, %C) : 
    (memref<1200000xf32>, memref<1320000xf32>, memref<1100000xf32>) -> ()
  %t_end = call @rtclock() : () -> f64
  %t = subf %t_end, %t_start : f64
  %num_flops = constant 2640000000 : index
  %num_flops_i = index_cast %num_flops : index to i64
  %num_flops_f = sitofp %num_flops_i : i64 to f64
  %flops = divf %num_flops_f, %t : f64
  call @print_flops(%flops) : (f64) -> ()
  //%pC = memref_cast %C : memref<1000x1100xf32> to memref<*xf32>
  //call @print_memref_f32(%pC) : (memref<*xf32>) -> ()
  return 
}

func @print_flops(f64)
func @rtclock() -> f64
