#include "mlir/Conversion/RaiseAffineToLinalg/RaiseAffineToLinalgPass.h"
#include "mlir/Analysis/NestedMatcher.h"
#include "mlir/Dialect/AffineOps/AffineOps.h"
#include "mlir/Dialect/Linalg/EDSC/Intrinsics.h"
#include "mlir/Dialect/Linalg/IR/LinalgOps.h"
#include "mlir/Dialect/StandardOps/Ops.h"
#include "mlir/IR/Access.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/Support/Debug.h"

using namespace mlir;

namespace {
class RaiseAffineToLinalgPass : public FunctionPass<RaiseAffineToLinalgPass> {
  void runOnFunction() override;
};
} // namespace

#define DEBUG_TYPE "mlir-raise-to-affine"

static llvm::cl::opt<bool>
    clEmitCall("emit-blas-call", llvm::cl::desc("directly emit blas calls."),
               llvm::cl::init(false));

namespace {

/// Class for matching a matmul op.
class MatMulMatcher : public OpRewritePattern<AffineForOp> {
public:
  using OpRewritePattern<AffineForOp>::OpRewritePattern;

  PatternMatchResult matchBody(Region &body, Value i, Value j, Value k,
                               Value &operandA, Value &operandB,
                               Value &operandC) const {
    using namespace mlir::matchers;
    {
      AccessPatternContext pctx(body.getContext());

      auto _i = m_Placeholder();
      auto _j = m_Placeholder();
      auto _k = m_Placeholder();

      auto _A = m_ArrayPlaceholder();
      auto _B = m_ArrayPlaceholder();
      auto _C = m_ArrayPlaceholder();

      auto a = m_Op<AffineLoadOp>(_A({_i, _k}));
      auto b = m_Op<AffineLoadOp>(_B({_k, _j}));
      auto c = m_Op<AffineLoadOp>(_C({_i, _j}));

      // TODO: relax the order of operands and introduce
      // ArrayPlaceholder.
      auto matMulOp = m_Op<AddFOp>(c, m_Op<MulFOp>(a, b));

      auto store = dyn_cast<AffineStoreOp>(*std::prev(body.front().end(), 2));

      if (!matchPattern(store, m_Op<AffineStoreOp>(_C({_i, _j}))))
        return matchFailure();

      auto add =
          dyn_cast_or_null<AddFOp>(store.getValueToStore().getDefiningOp());
      if (!add)
        return matchFailure();

      if (!matMulOp.match(add))
        return matchFailure();

      // TODO: we may be lenient to operations without side-effects, but they
      // should have been removed by DCE beforehand.
      if (std::distance(body.front().begin(), body.front().end()) != 7)
        return matchFailure();

      if ((i != pctx[_i]) || (j != pctx[_j]) || (k != pctx[_k]))
        return matchFailure();

      operandA = pctx[_A];
      operandB = pctx[_B];
      operandC = pctx[_C];
    }
    return matchSuccess();
  }

  PatternMatchResult
  matchAndRewriteNestedPattern(Operation *op, PatternRewriter &rewriter) const {
    Value operandA, operandB, operandC;
    auto body = [this, &operandA, &operandB, &operandC](Operation &op) -> bool {
      auto loop = cast<AffineForOp>(op);
      Value k = loop.getInductionVar();
      auto parent = loop.getParentOfType<AffineForOp>();
      Value j = parent.getInductionVar();
      parent = parent.getParentOfType<AffineForOp>();
      Value i = parent.getInductionVar();
      return matchBody(loop.getLoopBody(), i, j, k, operandA, operandB,
                       operandC)
          .hasValue();
    };

    {
      NestedPatternContext raii;
      using namespace matcher;
      auto m = For(For(For(body)));
      SmallVector<NestedMatch, 1> matches;
      m.match(op, &matches);
      if (matches.empty())
        return matchFailure();
    }

    if ((!operandA) || (!operandB) || (!operandC))
      return matchFailure();

    LLVM_DEBUG(llvm::dbgs() << "Matched Matmul pattern!\n");

    SmallVector<Value, 3> operands = {operandA, operandB, operandC};
    if (clEmitCall) {
      // emit blas call.
      auto module = op->getParentOfType<ModuleOp>();
      auto operandAType = operandA.getType();
      auto operandBType = operandB.getType();
      auto operandCType = operandC.getType();
      SmallVector<Type, 4> inputTypes{operandAType, operandBType, operandCType};
      auto libFnType = FunctionType::get(inputTypes, {}, rewriter.getContext());
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPoint(module.getBody(),
                                 std::prev(module.getBody()->end()));
      rewriter.create<FuncOp>(op->getLoc(), "myMatmul", libFnType,
                              ArrayRef<NamedAttribute>{});
      rewriter.setInsertionPoint(op);
      rewriter.create<CallOp>(op->getLoc(), "myMatmul", ArrayRef<Type>{},
                              operands);
    } else {
      // emit linalg.
      using namespace edsc;
      using namespace edsc::ops;
      OpBuilder builder(op->getContext());
      ScopedContext scop(builder, op->getLoc());
      auto matMulOp = linalg_matmul(makeValueHandles(operands));
      rewriter.clone(*matMulOp);
      matMulOp->erase();
    }
    rewriter.eraseOp(op);
    return matchSuccess();
  }

  // main rewriting function.
  PatternMatchResult matchAndRewrite(AffineForOp op,
                                     PatternRewriter &rewriter) const override {
    return matchAndRewriteNestedPattern(op, rewriter);
  }
};

} // end namespace

void RaiseAffineToLinalgPass::runOnFunction() {

  ConversionTarget target(getContext());
  target.addLegalDialect<mlir::linalg::LinalgDialect, StandardOpsDialect>();
  target.addLegalOp<ReturnOp, AllocOp, ModuleOp, ModuleTerminatorOp, FuncOp>();

  OwningRewritePatternList patterns;

  patterns.insert<MatMulMatcher>(&getContext());

  // run full conversion. As we discussed we want only
  // library calls.
  auto function = getFunction();
  if (failed(applyFullConversion(function, target, patterns)))
    signalPassFailure();
}

std::unique_ptr<OpPassBase<FuncOp>> mlir::createRaiseAffineToLinalgPass() {
  return std::make_unique<RaiseAffineToLinalgPass>();
}

static PassRegistration<RaiseAffineToLinalgPass>
    pass("raise-affine-to-linalg", "Raise affine to linalg ops");
