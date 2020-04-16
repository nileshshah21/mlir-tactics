#include "HelpersGpu.h"

namespace {
#include "TestTacticsBlasGpu.inc"
}

namespace {
struct TestTacticsBlasDriver
    : public mlir::PassWrapper<TestTacticsBlasDriver, mlir::FunctionPass> {
  void runOnFunction() override {
    mlir::OwningRewritePatternList patterns;
    populateWithGenerated(&getContext(), &patterns);
    applyPatternsAndFoldGreedily(getFunction(), patterns);
  }
};
} // end namespace

namespace mlir {
void registerTacticsTestBlasPassGpu() {
  mlir::PassRegistration<TestTacticsBlasDriver>(
      "test-tactics-blas-gpu", "Run test blas tactics for gpu");
}
} // end namespace mlir
