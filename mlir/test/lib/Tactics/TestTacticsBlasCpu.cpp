#include "HelpersCpu.h"

namespace {
#include "TestTacticsBlasCpu.inc"
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
void registerTacticsTestBlasPassCpu() {
  mlir::PassRegistration<TestTacticsBlasDriver>(
      "test-tactics-blas-cpu", "Run test blas tactics for cpu");
}
} // end namespace mlir
