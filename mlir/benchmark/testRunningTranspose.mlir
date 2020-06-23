#map0 = affine_map<(d0, d1, d2) -> (d0 * 32768 + d2 * 32 + d1)>
#map1 = affine_map<(d0, d1, d2) -> (d0, d2, d1)>
#map2 = affine_map<(d0, d1)[s0] -> (d0 * s0 + d1 * 32)>
#map3 = affine_map<(d0, d1, d2) -> (d0, d1)>
#map4 = affine_map<(d0, d1, d2) -> (d2)>
#map5 = affine_map<(d0, d1, d2) -> (d0 * 32768 + d2 + d1 * 32)>


module {
  func @contraction.abc.acd.db(%arg0: memref<32x1024x32xf32>, %arg1: memref<32x32x1024xf32>, %arg2: memref<1024x1024xf32>) {
    %cst = constant 1.000000e+00 : f32
    %0 = linalg.transpose %arg0 (d0, d1, d2) -> (d0, d2, d1) : memref<32x1024x32xf32>
    %1 = linalg.reshape %0 [#map3, #map4] : memref<32x32x1024xf32, #map0> into memref<?x1024xf32, #map2>
    %2 = linalg.reshape %arg1 [#map3, #map4] : memref<32x32x1024xf32> into memref<1024x1024xf32>
    %3 = alloc() : memref<f32>
    %4 = alloc() : memref<f32>
    linalg.fill(%4, %cst) : memref<f32>, f32
    linalg.fill(%3, %cst) : memref<f32>, f32
    linalg.matmul(%4, %3, %2, %arg2, %1) : memref<f32>, memref<f32>, memref<1024x1024xf32>, memref<1024x1024xf32>, memref<?x1024xf32, #map2>
    %5 = linalg.reshape %1 [#map3, #map4] : memref<?x1024xf32, #map2> into memref<32x32x1024xf32, #map0>
    %6 = linalg.transpose %5 (d0, d1, d2) -> (d0, d2, d1) : memref<32x32x1024xf32, #map0>
    linalg.copy(%6, %arg0) : memref<32x1024x32xf32, #map5>, memref<32x1024x32xf32>
    return
  }
  func @main() {
    %cst = constant 1.000000e+00 : f32
    %cst_0 = constant 2.000000e+00 : f32
    %0 = alloc() : memref<32x1024x32xf32>
    %1 = alloc() : memref<32x32x1024xf32>
    %2 = alloc() : memref<1024x1024xf32>
    linalg.fill(%1, %cst_0) : memref<32x32x1024xf32>, f32
    linalg.fill(%2, %cst_0) : memref<1024x1024xf32>, f32
    linalg.fill(%0, %cst) : memref<32x1024x32xf32>, f32
    call @start_timer() : () -> ()
    call @contraction.abc.acd.db(%0, %1, %2) : (memref<32x1024x32xf32>, memref<32x32x1024xf32>, memref<1024x1024xf32>) -> ()
    call @stop_timer() : () -> ()
    %pC = memref_cast %0 : memref<32x1024x32xf32> to memref<*xf32>
    call @print_memref_f32(%pC) : (memref<*xf32>) -> ()
    return
  }
  func @start_timer()
  func @stop_timer()
  func @print_memref_f32(memref<*xf32>)
}
