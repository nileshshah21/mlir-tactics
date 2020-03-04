#include "Helpers.h"

namespace {
#include "TestTacticsBlas.inc"
}

namespace {
struct TestTacticsBlasDriver
    : public mlir::FunctionPass<TestTacticsBlasDriver> {
  void runOnFunction() override {
    mlir::OwningRewritePatternList patterns;
    populateWithGenerated(&getContext(), &patterns);
    applyPatternsGreedily(getFunction(), patterns);
  }
};
} // end namespace

namespace mlir {
void registerTacticsTestBlasPass() {
  mlir::PassRegistration<TestTacticsBlasDriver>("test-tactics-blas",
                                                "Run test blas tactics");
}
} // end namespace mlir
