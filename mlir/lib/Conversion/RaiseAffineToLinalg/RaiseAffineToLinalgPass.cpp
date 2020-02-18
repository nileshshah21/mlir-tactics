#include "mlir/Conversion/RaiseAffineToLinalg/RaiseAffineToLinalgPass.h"
#include "mlir/Dialect/Linalg/IR/LinalgOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

using namespace mlir;

namespace {
class RaiseAffineToLinalgPass : public FunctionPass<RaiseAffineToLinalgPass> {
  void runOnFunction() override;
};
} // namespace

void RaiseAffineToLinalgPass::runOnFunction() {

  ConversionTarget target(getContext());
  target.addLegalDialect<mlir::linalg::LinalgDialect>();
  target.addLegalOp<ModuleOp, ModuleTerminatorOp, FuncOp>();

  OwningRewritePatternList patterns;

  // patterns.insert<MatMulMatcher>(&getContext());

  auto function = getFunction();
  if (failed(applyFullConversion(function, target, patterns)))
    signalPassFailure();
}

std::unique_ptr<OpPassBase<FuncOp>> mlir::createRaiseAffineToLinalgPass() {
  return std::make_unique<RaiseAffineToLinalgPass>();
}

static PassRegistration<RaiseAffineToLinalgPass>
    pass("raise-affine-to-linalg", "Raise affine to linalg ops");
