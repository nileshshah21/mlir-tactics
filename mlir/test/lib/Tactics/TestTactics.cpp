#include "mlir/Analysis/NestedMatcher.h"
#include "mlir/Dialect/AffineOps/AffineOps.h"
#include "mlir/Dialect/Linalg/EDSC/Intrinsics.h"
#include "mlir/Dialect/Linalg/IR/LinalgOps.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/Access.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

namespace {
#include "TestTactics.inc"
}

namespace {
struct TestTacticsDriver : public mlir::FunctionPass<TestTacticsDriver> {
  void runOnFunction() override {
    mlir::OwningRewritePatternList patterns;
    populateWithGenerated(&getContext(), &patterns);
    applyPatternsGreedily(getFunction(), patterns);
  }
};
} // end namespace

namespace mlir {
void registerTacticsTestPass() {
  mlir::PassRegistration<TestTacticsDriver>("test-tactics", "Run test tactics");
}
} // end namespace mlir
