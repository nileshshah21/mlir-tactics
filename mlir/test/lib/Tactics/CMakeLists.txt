set(LLVM_OPTIONAL_SOURCES
  TestTactics.cpp
)

set(LLVM_TARGET_DEFINITIONS TestTactics.td)
mlir_tablegen(TestTactics.inc -gen-tactics-defs)
add_public_tablegen_target(MLIRTestTacticsInc)

add_llvm_library(MLIRTestTactics
  TestTactics.cpp
  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/IR
)


add_dependencies(MLIRTestTactics
  MLIRTestTacticsInc
)

target_link_libraries(MLIRTestTactics
  MLIRDialect
  MLIRIR
  MLIRPass
  LLVMSupport
  MLIRAffineOps
)
