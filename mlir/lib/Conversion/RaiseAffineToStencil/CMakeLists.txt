add_llvm_library(MLIRRaiseAffineToStencilTransforms 
   RaiseAffineToStencilPass.cpp

ADDITIONAL_HEADER_DIRS  
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/IR

)

target_link_libraries(MLIRRaiseAffineToStencilTransforms 
  MLIRIR 
  MLIRPass 
  MLIRSupport 
  MLIRAffine
)
