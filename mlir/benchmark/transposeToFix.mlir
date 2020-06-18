#map0 = affine_map<(d0, d1, d2) -> (d2 * 32768 + d0 * 32 + d1)>
#map1 = affine_map<(d0, d1, d2) -> (d2, d0, d1)>
#map2 = affine_map<(d0, d1, d2) -> (d0)>
#map3 = affine_map<(d0, d1, d2) -> (d1, d2)>
#map4 = affine_map<(d0, d1) -> (d0 + d1 * 32768)>
#map5 = affine_map<(d0, d1, d2) -> (d0, d1)>
#map6 = affine_map<(d0, d1, d2) -> (d2)>
#map7 = affine_map<() -> ()>
#map8 = affine_map<(d0, d1) -> (d0, d1)>
#map9 = affine_map<() -> (0)>
#map10 = affine_map<() -> (1024)>
#map11 = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#map12 = affine_map<() -> (32)>


module {
  func @contraction.ab.acd.dbc(%arg0: memref<1024x1024xf32>, %arg1: memref<1024x32x32xf32>, %arg2: memref<32x1024x32xf32>) {
    %cst = constant 1.000000e+00 : f32
    %0 = linalg.transpose %arg2 (d0, d1, d2) -> (d2, d0, d1) : memref<32x1024x32xf32>
    %1 = linalg.reshape %arg1 [#map2, #map3] : memref<1024x32x32xf32> into memref<1024x1024xf32>
    %2 = linalg.reshape %0 [#map5, #map6] : memref<32x32x1024xf32, #map0> into memref<1024x1024xf32, #map4>
    %3 = alloc() : memref<f32>
    %4 = alloc() : memref<f32>
    affine.store %cst, %4[] : memref<f32>
    affine.store %cst, %3[] : memref<f32>
    affine.for %arg3 = 0 to 1024 {
      affine.for %arg4 = 0 to 1024 {
        affine.for %arg5 = 0 to 1024 {
          %5 = affine.load %arg0[%arg3, %arg4] : memref<1024x1024xf32>
          %6 = affine.load %4[] : memref<f32>
          %7 = mulf %6, %5 : f32
          %8 = affine.load %2[%arg5, %arg4] : memref<1024x1024xf32, #map4>
          %9 = affine.load %1[%arg3, %arg5] : memref<1024x1024xf32>
          %10 = mulf %9, %8 : f32
          %11 = affine.load %3[] : memref<f32>
          %12 = mulf %11, %10 : f32
          %13 = addf %7, %12 : f32
          affine.store %13, %arg0[%arg3, %arg4] : memref<1024x1024xf32>
        }
      }
    }
    return
  }
  func @main() {
    %cst = constant 1.000000e+00 : f32
    %cst_0 = constant 2.000000e+00 : f32
    %0 = alloc() {alignment = 64 : i64} : memref<1024x1024xf32>
    %1 = alloc() {alignment = 64 : i64} : memref<1024x32x32xf32>
    %2 = alloc() {alignment = 64 : i64} : memref<32x1024x32xf32>
    affine.for %arg0 = 0 to 1024 {
      affine.for %arg1 = 0 to 32 {
        affine.for %arg2 = 0 to 32 {
          affine.store %cst_0, %1[%arg0, %arg1, %arg2] : memref<1024x32x32xf32>
        }
      }
    }
    affine.for %arg0 = 0 to 32 {
      affine.for %arg1 = 0 to 1024 {
        affine.for %arg2 = 0 to 32 {
          affine.store %cst_0, %2[%arg0, %arg1, %arg2] : memref<32x1024x32xf32>
        }
      }
    }
    affine.for %arg0 = 0 to 1024 {
      affine.for %arg1 = 0 to 1024 {
        affine.store %cst, %0[%arg0, %arg1] : memref<1024x1024xf32>
      }
    }
    call @contraction.ab.acd.dbc(%0, %1, %2) : (memref<1024x1024xf32>, memref<1024x32x32xf32>, memref<32x1024x32xf32>) -> ()
    return
  }
}

