func @im2col(%s: memref<5x5xf32>, %f: memref<3x3xf32>) {
  %2 = alloc() : memref<9x9xf32> 
  linalg.imtocol %s %2 {block = [2, 2]} : memref<5x5xf32>, memref<9x9xf32>
  %3 = alloc() : memref<9x9xf32>
  linalg.copy (%2, %3) : memref<9x9xf32>, memref<9x9xf32>
  return
}
