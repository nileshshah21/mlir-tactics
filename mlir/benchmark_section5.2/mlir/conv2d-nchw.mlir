func @channelConv(%out: memref<256x256xf32>, %filt: memref<256x256xf32>, %img: memref<512x512xf32>) {
    affine.for %out_h = 0 to 256 {
      affine.for %out_w = 0 to 256 {
        affine.for %k_h = 0 to 256 {
          affine.for %k_w = 0 to 256 {
            %0 = affine.load %out[%out_h, %out_w] : memref<256x256xf32>
            %1 = affine.load %filt[%k_h, %k_w] : memref<256x256xf32>
            %2 = affine.load %img[%out_h + %k_h, %out_w + %k_w] : memref<512x512xf32>
            %3 = mulf %1, %2 : f32
            %4 = addf %0, %3 : f32
            affine.store %4, %out[%out_h, %out_w] : memref<256x256xf32>
          }
        }
      }
    }
  return
}

func @main() {
  %img = alloc() : memref<512x512xf32>
  %filt = alloc() : memref<256x256xf32>
  %dest = alloc() : memref<256x256xf32>

  %cf1 = constant 1.00000e+00 : f32

  linalg.fill(%img, %cf1) : memref<512x512xf32>, f32
  linalg.fill(%filt, %cf1) : memref<256x256xf32>, f32
  linalg.fill(%dest, %cf1) : memref<256x256xf32>, f32

  call @channelConv(%dest, %filt, %img) :
    (memref<256x256xf32>, memref<256x256xf32>, memref<512x512xf32>) -> ()
  %pc = memref_cast %dest : memref<256x256xf32> to memref<*xf32>
  call @print_memref_f32(%pc) : (memref<*xf32>) -> ()
  return
}

func @print_memref_f32(memref<*xf32>)
