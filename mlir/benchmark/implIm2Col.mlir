module {
  func @main() { 
    
    // img
    %1 = alloc() : memref<5x5xf32>
    %cf1 = constant 2.00000e+00 : f32
    linalg.fill(%1, %cf1) : memref<5x5xf32>, f32
    
    %img_reshaped = linalg.reshape %1 [affine_map<(i, j) -> (i, j)>] 
      : memref<5x5xf32> into memref<25xf32>
    
    // col
    %col_buffer = alloc() : memref<81xf32>

    %height_col = constant 3 : index
    %width_col = constant 3 : index
    %channel = constant 9 : index
    %kernel_w = constant 3 : i64
    %kernel_h = constant 3 : i64
    %stride_h = constant 1 : i64
    %stride_w = constant 1 : i64
    %pad_w = constant 0 : i64
    %pad_h = constant 0 : i64
    %height = constant 5 : i64
    %width = constant 5 : i64
    %zero = constant 0 : index

    affine.for %arg2 = 0 to %channel {
      %2 = index_cast %arg2 : index to i64
      %w_offset = remi_signed %2, %kernel_w : i64
      %3 = divi_signed %2, %kernel_w : i64
      %h_offset = remi_signed %3, %kernel_h : i64
      %4 = muli %kernel_h, %kernel_w : i64
      %c_im = divi_signed %2, %4 : i64
      affine.for %arg3 = 0 to %height_col {
        affine.for %arg4 = 0 to %width_col {
          %8 = index_cast %arg3 : index to i64
          %9 = index_cast %arg4 : index to i64
          %height_col_casted = index_cast %height_col : index to i64
          %width_col_casted = index_cast %width_col : index to i64
          %10 = muli %8, %stride_h : i64
          %11 = subi %10, %pad_h : i64
          %h_pad = addi %11, %h_offset : i64
          %12 = muli %9, %stride_w : i64
          %13 = subi %12, %pad_w : i64
          %w_pad = addi %13, %w_offset : i64
          %14 = muli %2, %height_col_casted : i64
          %15 = addi %14, %8 : i64
          %16 = muli %15, %width_col_casted : i64
          %col_index = addi %16, %9 : i64
          %17 = muli %c_im, %height : i64
          %18 = addi %17, %h_pad : i64
          %19 = muli %18, %width : i64
          %img_index = addi %19, %w_pad : i64
          %img_index_casted = index_cast %img_index : i64 to index
          %col_index_casted = index_cast %col_index : i64 to index
          
          %20 = load %img_reshaped[%img_index_casted] : memref<25xf32>
          store %20, %col_buffer[%col_index_casted] : memref<81xf32>
        }
      }
    }
    
    %pc = memref_cast %col_buffer : memref<81xf32> to memref<*xf32>
    call @print_memref_f32(%pc) : (memref<*xf32>) -> () 
    return
  }
func @print_memref_f32(memref<*xf32>)
}

