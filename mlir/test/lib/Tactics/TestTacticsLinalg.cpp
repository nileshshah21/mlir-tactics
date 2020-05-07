#include "HelpersLinalg.h"

namespace {
#include "TestTacticsLinalg.inc"
}

namespace {
struct TestTacticsLinalgDriver
    : public mlir::PassWrapper<TestTacticsLinalgDriver, mlir::FunctionPass> {
  void runOnFunction() override {
    mlir::OwningRewritePatternList patterns;
    populateWithGenerated(&getContext(), &patterns);
    applyPatternsAndFoldGreedily(getFunction(), patterns);
  }
};
} // end namespace

namespace mlir {
void registerTacticsTestLinalgPass() {
  mlir::PassRegistration<TestTacticsLinalgDriver>("test-tactics-linalg",
                                                  "Run test tactics");
}
} // end namespace mlir
