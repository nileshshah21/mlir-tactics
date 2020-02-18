#ifndef MLIR_CONVERSION_RAISE_AFFINE_TO_LINALG_PASS_H
#define MLIR_CONVERSION_RAISE_AFFINE_TO_LINALG_PASS_H

#include "mlir/Pass/Pass.h"

namespace mlir {

/// Creates and returns a pass to raise Affine to Linalg Ops.
std::unique_ptr<OpPassBase<FuncOp>> createRaiseAffineToLinalgPass();

} // end namespace mlir.

#endif
