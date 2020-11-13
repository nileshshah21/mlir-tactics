func @contraction.abc.bda.dc(%C: memref<32x32x1024xf32>, %A: memref<32x1024x32xf32>, %B: memref<1024x1024xf32>) {
  affine.for %a = 0 to 32 {
    affine.for %b = 0 to 32 {
      affine.for %c = 0 to 1024 {
        affine.for %d = 0 to 1024 {
          %0 = affine.load %A[%b, %d, %a] : memref<32x1024x32xf32>
          %1 = affine.load %B[%d, %c] : memref<1024x1024xf32>
          %2 = affine.load %C[%a, %b, %c] : memref<32x32x1024xf32>
          %3 = mulf %0, %1 : f32
          %4 = addf %2, %3 : f32
          affine.store %4, %C[%a, %b, %c] : memref<32x32x1024xf32>
        }
      }
    }
  }
  return
}

func @main() {
  %C = alloc() : memref<32x32x1024xf32>
  %A = alloc() : memref<32x1024x32xf32>
  %B = alloc() : memref<1024x1024xf32>

  %cf0 = constant 1.00000e+00 : f32
  %cf1 = constant 2.00000e+00 : f32

  linalg.fill(%A, %cf1) : memref<32x1024x32xf32>, f32
  linalg.fill(%B, %cf1) : memref<1024x1024xf32>, f32
  linalg.fill(%C, %cf0) : memref<32x32x1024xf32>, f32

  %t_start = call @rtclock() : () -> (f64)
  call @contraction.abc.bda.dc(%C, %A, %B) :
    (memref<32x32x1024xf32>, memref<32x1024x32xf32>, memref<1024x1024xf32>) -> ()
  %t_end = call @rtclock() : () -> (f64)
  %t = subf %t_end, %t_start : f64
  //%num_flops = constant 2147483648 : index
  //%num_flops_i = index_cast %num_flops : index to i64
  //%num_flops_f = sitofp %num_flops_i : i64 to f64
  //%flops = divf %num_flops_f, %t : f64
  //call @print_flops(%flops) : (f64) -> ()
  //%pC = memref_cast %C : memref<32x32x1024xf32> to memref<*xf32>
  //call @print_memref_f32(%pC) : (memref<*xf32>) -> ()
  call @print_double(%t) : (f64) -> ()
  return
}

func @rtclock() -> f64
func @print_flops(f64)
func @print_double(f64)
func @print_memref_f32(memref<*xf32>)
