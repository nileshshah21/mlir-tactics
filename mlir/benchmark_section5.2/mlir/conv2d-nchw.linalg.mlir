func @main() {
  %img = alloc() : memref<1x1x512x512xf32>
  %filt = alloc() : memref<1x1x256x256xf32>
  %dest = alloc() : memref<1x1x256x256xf32>

  %cf1 = constant 1.00000e+00 : f32

  linalg.fill(%img, %cf1) : memref<1x1x512x512xf32>, f32
  linalg.fill(%filt, %cf1) : memref<1x1x256x256xf32>, f32
  linalg.fill(%dest, %cf1) : memref<1x1x256x256xf32>, f32

  %t_start = call @rtclock() : () -> f64

  linalg.conv_2d_nchw %img, %filt, %dest :
    (memref<1x1x512x512xf32>, memref<1x1x256x256xf32>, memref<1x1x256x256xf32>)

  %t_end = call @rtclock() : () -> f64
  %t = subf %t_end, %t_start : f64
  call @print_double(%t) : (f64) -> ()
  return
}

func @print_double(f64)
func @rtclock() -> f64
