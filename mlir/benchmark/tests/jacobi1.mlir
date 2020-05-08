#map0 = affine_map<(d0) -> (d0 - 1)>
#map1 = affine_map<(d0) -> (d0)>
#map2 = affine_map<(d0) -> (d0 + 1)>
#map3 = affine_map<() -> (1)>
#map4 = affine_map<() -> (1999)>
#map5 = affine_map<() -> (0)>
#map6 = affine_map<() -> (500)>

func @scop_entry(%arg0: memref<2000xf32>, %arg1: memref<2000xf32>) {
    affine.for %arg2 = 0 to 500 {
      affine.for %arg3 = 1 to 1999 {
        %cst = constant 1.00000e+0 : f32
        %0 = affine.apply #map0(%arg3)
        %1 = affine.load %arg0[%0] : memref<2000xf32>
        %2 = affine.apply #map1(%arg3)
        %3 = affine.load %arg0[%2] : memref<2000xf32>
        %4 = addf %1, %3 : f32
        %5 = affine.apply #map2(%arg3)
        %6 = affine.load %arg0[%5] : memref<2000xf32>
        %7 = addf %4, %6 : f32
        %8 = mulf %cst, %7 : f32
        affine.store %8, %arg1[%arg3] : memref<2000xf32>
      }
    }
    return
}

func @main() {
  %A = alloc() : memref<2000xf32>
  %B = alloc() : memref<2000xf32>

  %cf1 = constant 1.00000e+00 : f32
  
  linalg.fill(%A, %cf1) : memref<2000xf32>, f32
  linalg.fill(%B, %cf1) : memref<2000xf32>, f32
  
  call @scop_entry(%A, %B) : (memref<2000xf32>, memref<2000xf32>) -> ()
  %pB = memref_cast %B : memref<2000xf32> to memref<*xf32>
  call @print_memref_f32(%pB) : (memref<*xf32>) -> ()  
  return
}

func @print_memref_f32(memref<*xf32>)
