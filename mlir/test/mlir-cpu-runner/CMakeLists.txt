# for cudaToolkit
cmake_minimum_required(VERSION 3.17.0)

set(LLVM_OPTIONAL_SOURCES
  mlir_test_cblas.cpp
  mlir_test_cblas_interface.cpp
  )

add_llvm_library(mlir_test_cblas SHARED mlir_test_cblas.cpp)
target_compile_definitions(mlir_test_cblas PRIVATE mlir_test_cblas_EXPORTS)


add_llvm_library(mlir_test_cblas_interface SHARED mlir_test_cblas_interface.cpp)
target_link_libraries(mlir_test_cblas_interface PRIVATE mlir_test_cblas)
target_compile_definitions(mlir_test_cblas_interface PRIVATE mlir_test_cblas_interface_EXPORTS)

find_package(MKLDNN REQUIRED)
include_directories(${MKLDNN_INCLUDE_PATH})
target_link_libraries(mlir_test_cblas_interface PRIVATE -L${MKLDNN_LIBRARY_PATH} dnnl)

find_package(CUDAToolkit)
if (CUDAToolkit_FOUND) 
  # Verify that all the CUDA:: targets and variables exist
  foreach (cuda_lib cudart cuda_driver cublas cufft cufftw curand cusolver cusparse nvgraph)
    if(NOT CUDA_${cuda_lib}_LIBRARY)
      message(FATAL_ERROR "expected CUDAToolkit variable CUDA_${cuda_lib}_LIBRARY not found")
    endif()
    if(NOT TARGET CUDA::${cuda_lib})
      message(FATAL_ERROR "expected CUDAToolkit target CUDA::${cuda_lib} not found")
    endif()
  endforeach()
  # include and link
  include_directories(${CUDAToolkit_INCLUDE_DIRS})
  target_link_libraries(mlir_test_cblas_interface PRIVATE CUDA::cublas)
  # add def
  add_definitions(-DHAS_CUBLAS)
endif()

