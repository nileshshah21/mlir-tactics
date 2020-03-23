set(LLVM_OPTIONAL_SOURCES
  TestTacticsLinalg.cpp
  TestTacticsBlasCpu.cpp
)

set(LLVM_TARGET_DEFINITIONS TestTactics.td)
mlir_tablegen(TestTacticsLinalg.inc -gen-tactics-linalg)
mlir_tablegen(TestTacticsBlasCpu.inc -gen-tactics-blas-cpu)
add_public_tablegen_target(MLIRTestTacticsLinalgInc)
add_public_tablegen_target(MLIRTestTacticsBlasCpuInc)

add_llvm_library(MLIRTestTacticsLinalg
  TestTacticsLinalg.cpp
  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/IR
)

add_llvm_library(MLIRTestTacticsBlasCpu
  TestTacticsBlasCpu.cpp
  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/IR
)

add_dependencies(MLIRTestTacticsBlasCpu
  MLIRTestTacticsBlasCpuInc
)

add_dependencies(MLIRTestTacticsLinalg
  MLIRTestTacticsLinalgInc
)

target_link_libraries(MLIRTestTacticsBlasCpu
  MLIRDialect
  MLIRIR
  MLIRPass
  LLVMSupport
  MLIRAffine
) 

target_link_libraries(MLIRTestTacticsLinalg
  MLIRDialect
  MLIRIR
  MLIRPass
  LLVMSupport
  MLIRAffine
)
