#include "HelpersGpu.h"

namespace {
#include "TestTacticsBlasGpu.inc"
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
void registerTacticsTestBlasPassGpu() {
  mlir::PassRegistration<TestTacticsBlasDriver>(
      "test-tactics-blas-gpu", "Run test blas tactics for gpu");
}
} // end namespace mlir
