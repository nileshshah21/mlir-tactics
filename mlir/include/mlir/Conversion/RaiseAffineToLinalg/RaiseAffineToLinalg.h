#ifndef MLIR_CONVERSION_RAISE_AFFINE_TO_LINALG_H
#define MLIR_CONVERSION_RAISE_AFFINE_TO_LINALG_H

namespace mlir {

class MLIRContext;
class OwningRewritePatternList;

void populateAffineToStdConversionPatterns(OwningRewritePatternList &patterns,
                                           MLIRContext *ctx);

} // end namespace mlir.

#endif
