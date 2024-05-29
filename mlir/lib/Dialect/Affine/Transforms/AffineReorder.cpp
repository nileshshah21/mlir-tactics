#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/Passes.h"
#include "mlir/Dialect/Affine/Utils.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
#include "PassDetail.h"
#include "mlir/Analysis/LoopAnalysis.h"
#include "mlir/Dialect/Affine/Passes.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/Builders.h"
#include "mlir/Transforms/LoopUtils.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"

using namespace mlir; 

namespace {
struct AffineReorder : public AffineReorderBase<AffineReorder> {

public:
  AffineReorder() = default;
  AffineReorder(const AffineReorder &pass) = default;

private:
  void runOnFunction() override {
    // Print affine load operations before the transformation
    // getOperation().walk([&](mlir::AffineLoadOp op) {
    //   llvm::errs() << "AffineLoadOp: " << op << "\n";
    // });
    getOperation().walk([&](AffineForOp loopOp) {
      // Look for innermost loops
      if (loopOp.getBody()->getOperations().size() == 1 &&
          isa<AffineForOp>(loopOp.getBody()->front())) {
        return; // Not the innermost loop
      }

      AffineLoadOp firstLoadOp = nullptr;
      Operation *firstUseAfterLoad = nullptr;

      for (Operation &op : loopOp.getBody()->getOperations()) {
        if (isa<AffineLoadOp>(op) && !firstLoadOp) {
          firstLoadOp = cast<AffineLoadOp>(&op);
          continue;
        }
        // Identify the first operation that uses the value loaded by
        // firstLoadOp
        for (const auto &operand : op.getOperands()) {
          if (operand.getDefiningOp() == firstLoadOp.getOperation()) {
            firstUseAfterLoad = &op;
            break;
          }
        }
        if (firstUseAfterLoad)
          break; // Stop searching once we find the first use
      }

      if (firstLoadOp && firstUseAfterLoad) {
        // Move the load just before its first use
        firstLoadOp.getOperation()->moveBefore(firstUseAfterLoad);
      }
    });
  }
}; // namespace
} // namespace

std::unique_ptr<OperationPass<FuncOp>>
mlir::createAffineReorderPass() {
  return std::make_unique<AffineReorder>();
}