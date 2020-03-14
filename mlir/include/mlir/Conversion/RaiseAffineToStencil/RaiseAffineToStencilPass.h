#ifndef MLIR_CONVERSION_RAISE_AFFINE_TO_STENCIL_PASS_H
#define MLIR_CONVERSION_RAISE_AFFINE_TO_STENCIL_PASS_H

#include "mlir/Pass/Pass.h"

namespace mlir {

/// Creates and returns a pass to raise Affine to Stencil Ops.
std::unique_ptr<OpPassBase<FuncOp>> createRaiseAffineToStencilPass();

} // end namespace mlir.

#endif
