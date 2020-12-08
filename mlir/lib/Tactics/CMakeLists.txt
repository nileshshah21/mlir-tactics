set(LLVM_OPTIONAL_SOURCES
  TacticsLinalg.cpp
  TacticsBlasCpu.cpp
  TacticsBlasGpu.cpp
)

set(LLVM_TARGET_DEFINITIONS Tactics.td)
mlir_tablegen(TacticsLinalg.inc -gen-tactics-linalg)
mlir_tablegen(TacticsBlasCpu.inc -gen-tactics-blas-cpu)
mlir_tablegen(TacticsBlasGpu.inc -gen-tactics-blas-gpu)
add_public_tablegen_target(MLIRTacticsLinalgInc)
add_public_tablegen_target(MLIRTacticsBlasCpuInc)
add_public_tablegen_target(MLIRTacticsBlasGpuInc)

add_llvm_library(MLIRTacticsLinalg
  TacticsLinalg.cpp
  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/IR
)

add_llvm_library(MLIRTacticsBlasCpu
  TacticsBlasCpu.cpp
  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/IR
)

add_llvm_library(MLIRTacticsBlasGpu
  TacticsBlasGpu.cpp
  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/IR
)

add_dependencies(MLIRTacticsBlasCpu
  MLIRTacticsBlasCpuInc
)

add_dependencies(MLIRTacticsLinalg
  MLIRTacticsLinalgInc
)

add_dependencies(MLIRTacticsBlasGpu
  MLIRTacticsBlasGpuInc
)

target_link_libraries(MLIRTacticsBlasCpu
  MLIRDialect
  MLIRIR
  MLIRPass
  LLVMSupport
  MLIRAffineOps
) 

target_link_libraries(MLIRTacticsLinalg
  MLIRDialect
  MLIRIR
  MLIRPass
  LLVMSupport
  MLIRAffineOps
)

target_link_libraries(MLIRTacticsBlasGpu
  MLIRDialect
  MLIRIR
  MLIRPass
  LLVMSupport
  MLIRAffineOps
)
