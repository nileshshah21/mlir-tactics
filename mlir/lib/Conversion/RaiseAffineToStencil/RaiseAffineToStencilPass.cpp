#include "mlir/Conversion/RaiseAffineToStencil/RaiseAffineToStencilPass.h"
#include "mlir/Analysis/NestedMatcher.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/Access.h"
#include "mlir/IR/Matchers.h"
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

namespace {

class JacobiMatcher : public OpRewritePattern<AffineForOp> {
public:
  using OpRewritePattern<AffineForOp>::OpRewritePattern;

  LogicalResult matchBody(Region &body, Value i, Value &operandA,
                          Value &operandB) const {
    using namespace matchers;
    {
      AccessPatternContext pctx(body.getContext());

      auto _i = m_Placeholder();
      auto _A = m_ArrayPlaceholder();
      auto _B = m_ArrayPlaceholder();

      auto a = m_Op<AffineLoadOp>(_A({_i}));
      auto aPlusOne = m_Op<AffineLoadOp>(_A({_i + 1}));
      auto aMinusOne = m_Op<AffineLoadOp>(_A({_i - 1}));
      auto b = m_Op<AffineStoreOp>(_B({_i}));

      auto stencil = m_Op<MulFOp>(
          m_Op<AddFOp>(m_Op<AddFOp>(aPlusOne, a), aMinusOne), m_Constant());

      auto store = dyn_cast<AffineStoreOp>(*std::prev(body.front().end(), 2));
      if (!matchPattern(store, b))
        return failure();

      auto mul =
          dyn_cast_or_null<MulFOp>(store.getValueToStore().getDefiningOp());
      if ((!mul) || (!stencil.match(mul)))
        return failure();

      if (std::distance(body.front().begin(), body.front().end()) != 8)
        return failure();

      if (i != pctx[_i])
        return failure();

      operandA = pctx[_A];
      operandB = pctx[_B];
    }
    return success();
  }

  LogicalResult matchAndRewriteNestedPattern(Operation *op,
                                             PatternRewriter &rewriter) const {
    Value operandA, operandB;
    auto body = [this, &operandA, &operandB](Operation &op) -> bool {
      auto loop = cast<AffineForOp>(op);
      Value i = loop.getInductionVar(); // innermost loop - i
      return succeeded(matchBody(loop.getLoopBody(), i, operandA, operandB));
    };

    {
      // look for a 2d nested loop.
      NestedPatternContext raii;
      using namespace matcher;
      auto m = For(For(body));
      SmallVector<NestedMatch, 1> matches;
      m.match(op, &matches);
      if (matches.empty())
        return failure();
    }

    if ((!operandA) || (!operandB))
      return failure();

    LLVM_DEBUG(llvm::dbgs() << "Matched Jacobi1d pattern!\n");

    // cut
    rewriter.eraseOp(op);
    return success();
  }

  // main rewriting function.
  LogicalResult matchAndRewrite(AffineForOp op,
                                PatternRewriter &rewriter) const override {
    return matchAndRewriteNestedPattern(op, rewriter);
  }
};

} // end namespace

void RaiseAffineToStencilPass::runOnFunction() {

  ConversionTarget target(getContext());
  target.addLegalDialect<StandardOpsDialect>();
  target.addLegalOp<ReturnOp, AllocOp, ModuleOp, ModuleTerminatorOp, FuncOp>();

  OwningRewritePatternList patterns;
  patterns.insert<JacobiMatcher>(&getContext());

  auto function = getFunction();
  if (failed(applyPartialConversion(function, target, patterns)))
    signalPassFailure();
}

std::unique_ptr<OpPassBase<FuncOp>> mlir::createRaiseAffineToStencilPass() {
  return std::make_unique<RaiseAffineToStencilPass>();
}

static PassRegistration<RaiseAffineToStencilPass>
    pass("raise-affine-to-stencil", "Raise affine to stencil ops");
