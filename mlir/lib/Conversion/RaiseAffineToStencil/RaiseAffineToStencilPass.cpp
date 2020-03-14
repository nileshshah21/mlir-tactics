#include "mlir/Conversion/RaiseAffineToStencil/RaiseAffineToStencilPass.h"
#include "mlir/Dialect/AffineOps/AffineOps.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/Support/Debug.h"

using namespace mlir;

namespace {
class RaiseAffineToStencilPass : public FunctionPass<RaiseAffineToStencilPass> {
  void runOnFunction() override;
};
} // end namespace

#define DEBUG_TYPE "mlir-raise-to-stencil"

namespace {} // end namespace

void RaiseAffineToStencilPass::runOnFunction() {

  ConversionTarget target(getContext());
  target.addLegalDialect<StandardOpsDialect>();
  target.addLegalOp<ReturnOp, AllocOp, ModuleOp, ModuleTerminatorOp, FuncOp>();

  OwningRewritePatternList patterns;
  // patterns.insert<JacobiMatcher>(&getContext());

  auto function = getFunction();
  if (failed(applyPartialConversion(function, target, patterns)))
    signalPassFailure();
}

std::unique_ptr<OpPassBase<FuncOp>> mlir::createRaiseAffineToStencilPass() {
  return std::make_unique<RaiseAffineToStencilPass>();
}

static PassRegistration<RaiseAffineToStencilPass>
    pass("raise-affine-to-stencil", "Raise affine to stencil ops");
