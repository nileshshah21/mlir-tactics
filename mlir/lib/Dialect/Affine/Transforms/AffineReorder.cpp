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
#include "mlir/Support/LLVM.h"

using namespace mlir; 

namespace {
struct AffineReorder : public AffineReorderBase<AffineReorder> {

public:
  AffineReorder() = default;
  AffineReorder(const AffineReorder &pass) = default;

private:
  void runOnFunction() override { 
    // Move the load operation just before its first use
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

    // Reorder the load ops
    getOperation().walk([&](AffineForOp loopOp) -> WalkResult {
 
        // Collect all load operations
        llvm::errs() << "Processing affine.for operation:\n" << loopOp << "\n";

        SmallVector<AffineLoadOp, 16> loadOps;
        loopOp.walk([&](Operation *op) {
            if (auto loadOp = dyn_cast<AffineLoadOp>(op)) {
                loadOps.push_back(loadOp);
            }
        });

        // Get the block inside the affine.for operation
        Block &entryBlock = loopOp.getRegion().front();

        // Move all load operations to the beginning of the affine.for loop body
        for (AffineLoadOp loadOp : llvm::reverse(loadOps)) {
        // Check if the load operation can be safely moved
        bool canMove = true;
            for (auto operand : loadOp.getOperands()) {
            if (auto definingOp = operand.getDefiningOp()) {
                    if (definingOp->getBlock() != &entryBlock || !definingOp->isBeforeInBlock(loadOp)) {
                        canMove = false;
                        break;
                    }
                }
            }
            if (canMove) {
                loadOp.getOperation()->moveBefore(&entryBlock, entryBlock.getOperations().begin());
            }
        }

        // Ensure operands are in increasing order
        entryBlock.walk([&](Operation *op) {

            // Skip AffineLoadOps and AffineStoreOps and AffineYieldOps
            if (isa<AffineLoadOp>(op) || isa<AffineStoreOp>(op) || isa<AffineYieldOp>(op))
                return;

            llvm::errs() << "Processing operation:\n" << *op << "\n";

            //Sort operands for the current operation if needed
            for (unsigned i = 0, e = op->getNumOperands(); i < e - 1; ++i) { 
                op->getOperand(i).getLoc().dump();
                op->getOperand(i + 1).getLoc().dump();
                LocationAttr locAttr = op->getOperand(i).getLoc().dyn_cast<LocationAttr>();
                LocationAttr locAttr2 = op->getOperand(i + 1).getLoc().dyn_cast<LocationAttr>();;
                if (!locAttr) {
                    llvm::errs() << "Error in line number:\n" << "\n";
                }
                if (!locAttr2) {
                    llvm::errs() << "Error in line number:\n" << "\n";
                }
                FileLineColLoc fileLoc = locAttr.dyn_cast<FileLineColLoc>();
                FileLineColLoc fileLoc2 = locAttr2.dyn_cast<FileLineColLoc>();
                // Check if the attribute contains a FileLineColLoc
                if (fileLoc) {
                    llvm::errs() << "Line number:\n" << fileLoc.getLine() << "\n"; ;
                }
                if (fileLoc2) {
                    llvm::errs() << "Line number:\n" << fileLoc2.getLine() << "\n"; ;
                }
                if (fileLoc.getLine() > fileLoc2.getLine()) {
                    // Swap operands to maintain the increasing order of line numbers
                    auto temp = op->getOperand(i);
                    op->setOperand(i, op->getOperand(i + 1));
                    op->setOperand(i + 1, temp);
                }
            }
        });

        return WalkResult::interrupt();
        
    });

        
  }
}; // namespace
} // namespace

std::unique_ptr<OperationPass<FuncOp>>
mlir::createAffineReorderPass() {
  return std::make_unique<AffineReorder>();
}