#map0 = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<() -> (0)>
#map2 = affine_map<(d0) -> (d0)>
#map3 = affine_map<() -> (1200)>
#map4 = affine_map<() -> (1000)>

func @scop_entry(%arg0: memref<1200x1000xf32>, 
                 %arg1: memref<1200x1200xf32>, %arg2: f32, %arg3: f32) {
    affine.for %arg4 = 0 to 1200 {
      affine.for %arg5 = 0 to #map2(%arg4) {
        %0 = affine.load %arg1[%arg4, %arg5] : memref<1200x1200xf32>
        %1 = mulf %arg3, %0 : f32
        affine.store %1, %arg1[%arg4, %arg5] : memref<1200x1200xf32>
      }
    }
    affine.for %arg4 = 0 to 1200 {
      affine.for %arg5 = 0 to 1000 {
        affine.for %arg6 = 0 to #map2(%arg4) {
          %0 = affine.load %arg0[%arg4, %arg5] : memref<1200x1000xf32>
          %1 = mulf %arg2, %0 : f32
          %2 = affine.load %arg0[%arg6, %arg5] : memref<1200x1000xf32>
          %3 = mulf %1, %2 : f32
          %4 = affine.load %arg1[%arg4, %arg6] : memref<1200x1200xf32>
          %5 = addf %3, %4 : f32
          affine.store %5, %arg1[%arg4, %arg6] : memref<1200x1200xf32>
        }
      }
    }
    return
}

func @main() {
  %A = alloc() : memref<1200x1000xf32>
  %C = alloc() : memref<1200x1200xf32>
  
  %cf1 = constant 1.00000e+00 : f32
  %cf2 = constant 2.00000e+00 : f32

  linalg.fill(%A, %cf1) : memref<1200x1000xf32>, f32
  linalg.fill(%C, %cf2) : memref<1200x1200xf32>, f32

  call @start_timer() : () -> ()
  call @scop_entry(%A, %C, %cf1, %cf2) :
    (memref<1200x1000xf32>, memref<1200x1200xf32>, f32, f32) -> ()
  call @stop_timer() : () -> ()
  return
}

func @start_timer()
func @stop_timer()
