add_llvm_library(MLIRRaiseAffineToLinalgTransforms 
   RaiseAffineToLinalgPass.cpp

ADDITIONAL_HEADER_DIRS 
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Dialect/Linalg 
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/IR

)

target_link_libraries(MLIRRaiseAffineToLinalgTransforms 
  MLIRIR 
  MLIRLinalgOps 
  MLIRLinalgUtils
  MLIRPass 
  MLIRSupport 
  MLIRAffineOps
)
