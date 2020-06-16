#include "PassDetail.h"
#include "mlir/Analysis/AffineAnalysis.h"
#include "mlir/Analysis/AffineStructures.h"
#include "mlir/Analysis/LoopAnalysis.h"
#include "mlir/Analysis/NestedMatcher.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/IR/Access.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/Passes.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "blis-opt"

using namespace mlir;

static void moveLoopBody(AffineForOp src, AffineForOp dest) {
  auto &inst = src.getBody()->getOperations();
  dest.getBody()->getOperations().splice(dest.getBody()->begin(), inst,
                                         inst.begin(), std::prev(inst.end()));
}

static bool alreadyFired(int atStep, Operation &op) {
  if (auto attr = op.getAttr("BLIS"))
    if (auto stringAttr = attr.dyn_cast<StringAttr>()) {
      auto step = std::stoi((stringAttr.getValue().str()));
      if (step >= atStep)
        return true;
    }
  return false;
}

static LogicalResult
interchangeBands(PatternRewriter &rewriter,
                 MutableArrayRef<AffineForOp> origLoops,
                 SmallVector<AffineForOp, 3> &interchangedLoops, int step,
                 int posInnermost) {
  auto rootFor = origLoops[0].getOperation();
  auto loc = rootFor->getLoc();
  auto depth = origLoops.size();
  AffineForOp innermost;
  for (size_t i = 0; i < depth; i++) {
    auto loop = rewriter.create<AffineForOp>(
        loc, origLoops[i].getLowerBoundOperands(),
        origLoops[i].getLowerBoundMap(), origLoops[i].getUpperBoundOperands(),
        origLoops[i].getUpperBoundMap(), origLoops[i].getStep());
    loop.getBody()->clear();
    rewriter.setInsertionPointToStart(loop.getBody());
    rewriter.create<AffineTerminatorOp>(loc);
    rewriter.setInsertionPointToStart(loop.getBody());
    auto loopAsOp = loop.getOperation();
    loopAsOp->setAttr(
        "BLIS", StringAttr::get(std::to_string(step), rewriter.getContext()));
    interchangedLoops.push_back(loop);
    if (i == depth - 1)
      innermost = loop;
  }
  moveLoopBody(origLoops[posInnermost], innermost);
  SmallVector<Value, 4> origLoopsIVs;
  extractForInductionVars(origLoops, &origLoopsIVs);
  for (size_t i = 0; i < depth; i++)
    origLoopsIVs[i].replaceAllUsesWith(interchangedLoops[i].getInductionVar());
  return success();
}

void setTiledIndexesHyperRect(PatternRewriter &rewriter,
                              MutableArrayRef<AffineForOp> origLoops,
                              MutableArrayRef<AffineForOp> newLoops,
                              ArrayRef<unsigned> tileSizes) {
  auto depth = origLoops.size();
  // Bounds for tile space loops.
  for (size_t i = 0; i < depth; i++) {
    OperandRange newLbOperands = origLoops[i].getLowerBoundOperands();
    OperandRange newUbOperands = origLoops[i].getUpperBoundOperands();
    newLoops[i].setLowerBound(newLbOperands, origLoops[i].getLowerBoundMap());
    newLoops[i].setUpperBound(newUbOperands, origLoops[i].getUpperBoundMap());
    newLoops[i].setStep(tileSizes[i]);
  }
  // Bounds for intra-tile loops.
  for (unsigned i = 0; i < depth; i++) {
    int64_t largestDiv = getLargestDivisorOfTripCount(origLoops[i]);
    auto mayBeConstantCount = getConstantTripCount(origLoops[i]);
    // The lower bound is just the tile-space loop.
    AffineMap lbMap = rewriter.getDimIdentityMap();
    newLoops[depth + i].setLowerBound(newLoops[i].getInductionVar(), lbMap);

    // Set the upper bound.
    if (mayBeConstantCount && mayBeConstantCount.getValue() < tileSizes[i]) {
      // Trip count is less than the tile size: upper bound is lower bound +
      // trip count.
      auto ubMap =
          rewriter.getSingleDimShiftAffineMap(mayBeConstantCount.getValue());
      newLoops[depth + i].setUpperBound(newLoops[i].getInductionVar(), ubMap);
    } else if (largestDiv % tileSizes[i] != 0) {
      // Intra-tile loop ii goes from i to min(i + tileSize, ub_i).
      // Construct the upper bound map; the operands are the original operands
      // with 'i' (tile-space loop) appended to it. The new upper bound map is
      // the original one with an additional expression i + tileSize appended.

      // Add dim operands from original upper bound.
      SmallVector<Value, 4> ubOperands;
      auto ub = origLoops[i].getUpperBound();
      ubOperands.reserve(ub.getNumOperands() + 1);
      auto origUbMap = ub.getMap();
      for (unsigned j = 0, e = origUbMap.getNumDims(); j < e; ++j)
        ubOperands.push_back(ub.getOperand(j));

      // Add dim operand for new loop upper bound.
      ubOperands.push_back(newLoops[i].getInductionVar());

      // Add symbol operands from original upper bound.
      for (unsigned j = 0, e = origUbMap.getNumSymbols(); j < e; ++j)
        ubOperands.push_back(ub.getOperand(origUbMap.getNumDims() + j));

      SmallVector<AffineExpr, 4> boundExprs;
      boundExprs.reserve(1 + origUbMap.getNumResults());
      auto dim = rewriter.getAffineDimExpr(origUbMap.getNumDims());
      // The new upper bound map is the original one with an additional
      // expression i + tileSize appended.
      boundExprs.push_back(dim + tileSizes[i]);
      boundExprs.append(origUbMap.getResults().begin(),
                        origUbMap.getResults().end());
      auto ubMap =
          AffineMap::get(origUbMap.getNumDims() + 1, origUbMap.getNumSymbols(),
                         boundExprs, rewriter.getContext());
      newLoops[depth + i].setUpperBound(ubOperands, ubMap);
    } else {
      // No need of the min expression.
      auto dim = rewriter.getAffineDimExpr(0);
      auto ubMap = AffineMap::get(1, 0, dim + tileSizes[i]);
      newLoops[depth + i].setUpperBound(newLoops[i].getInductionVar(), ubMap);
    }
  }
}

LogicalResult
tilePerfectlyNestedBands(int step, PatternRewriter &rewriter,
                         MutableArrayRef<AffineForOp> input,
                         ArrayRef<unsigned> tileSizes,
                         SmallVectorImpl<AffineForOp> *tiledNest = nullptr) {
  auto rootFor = input[0].getOperation();
  auto loc = rootFor->getLoc();
  auto depth = input.size();

  AffineForOp innermost;
  SmallVector<AffineForOp, 6> tiledLoops(2 * depth);
  for (size_t i = 0; i < depth * 2; i++) {
    auto loop = rewriter.create<AffineForOp>(loc, 0, 0);
    loop.getBody()->clear();
    rewriter.setInsertionPointToStart(loop.getBody());
    rewriter.create<AffineTerminatorOp>(loc);
    rewriter.setInsertionPointToStart(loop.getBody());
    auto loopAsOp = loop.getOperation();
    loopAsOp->setAttr(
        "BLIS", StringAttr::get(std::to_string(step), rewriter.getContext()));
    tiledLoops[i] = loop;
    if (i == depth * 2 - 1)
      innermost = loop;
  }

  moveLoopBody(input.back(), innermost);

  SmallVector<Value, 4> origLoopIVs;
  extractForInductionVars(input, &origLoopIVs);

  FlatAffineConstraints cst;
  getIndexSet(input, &cst);
  cst.dump();
  if (!cst.isHyperRectangular(0, depth)) {
    llvm::dbgs() << "tiled code generation unimplemented for the "
                    "non-hyperrectangular case, op:"
                 << *rootFor << "\n";
    return failure();
  }
  setTiledIndexesHyperRect(rewriter, input, tiledLoops, tileSizes);
  // Replace original IVs with intra-tile loop IVs.
  for (unsigned i = 0; i < depth; i++)
    origLoopIVs[i].replaceAllUsesWith(tiledLoops[i + depth].getInductionVar());

  return success();
}

static LogicalResult isGemmNN(Region &body, Value i, Value j, Value k) {
  using namespace matchers;
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

    auto matMulOp = m_Op<AddFOp>(c, m_Op<MulFOp>(a, b));

    auto store = dyn_cast<AffineStoreOp>(*std::prev(body.front().end(), 2));

    if (!matchPattern(store, m_Op<AffineStoreOp>(_C({_i, _j}))))
      return failure();

    auto add =
        dyn_cast_or_null<AddFOp>(store.getValueToStore().getDefiningOp());
    if ((!add) || (!matMulOp.match(add)))
      return failure();

    if (std::distance(body.front().begin(), body.front().end()) != 7)
      return failure();

    if ((i != pctx[_i]) || (j != pctx[_j]) || (k != pctx[_k]))
      return failure();
  }
  return success();
}

///
class BlisMatmulOpStepTwo : public RewritePattern {
public:
  BlisMatmulOpStepTwo(MLIRContext *context)
      : RewritePattern(AffineForOp::getOperationName(), 0, context) {}

  LogicalResult matchBody(Region &body, Value i, Value j, Value k) const {
    return isGemmNN(body, i, j, k);
  };

  LogicalResult matchAndRewriteNestedPattern(Operation *op,
                                             PatternRewriter &rewriter) const {
    AffineForOp loopI, loopJ, loopK = nullptr;
    auto body = [this, &loopI, &loopJ, &loopK](Operation &op) -> bool {
      // exit if this stage already fired.
      if (auto attr = op.getAttr("BLIS"))
        if (auto stringAttr = attr.dyn_cast<StringAttr>()) {
          auto step = std::stoi((stringAttr.getValue().str()));
          if (step >= 2)
            return false;
        }
      auto loop = cast<AffineForOp>(op);
      loopI = loop;
      auto i = loop.getInductionVar();
      auto parent = loop.getParentOfType<AffineForOp>();
      loopK = parent;
      auto k = parent.getInductionVar();
      parent = parent.getParentOfType<AffineForOp>();
      loopJ = parent;
      auto j = parent.getInductionVar();
      return succeeded(matchBody(loop.getLoopBody(), i, j, k));
    };
    {
      NestedPatternContext raii;
      using namespace matcher;
      auto m = For(For(For(body)));
      SmallVector<NestedMatch, 1> matches;
      m.match(op, &matches);
      if (matches.empty())
        return failure();
    }
    LLVM_DEBUG(llvm::dbgs() << "Matched Matmul pattern!\n");
    SmallVector<AffineForOp, 3> loops = {loopJ, loopK, loopI};
    SmallVector<unsigned, 6> tileSizes = {32, 32, 32};
    if (failed(tilePerfectlyNestedBands(2, rewriter, loops, tileSizes)))
      return failure();
    return success();
  };

  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    return matchAndRewriteNestedPattern(op, rewriter);
  };
};

///
class BlisMatmulOpStepOne : public RewritePattern {
public:
  BlisMatmulOpStepOne(MLIRContext *context)
      : RewritePattern(AffineForOp::getOperationName(), 0, context) {}

  LogicalResult matchBody(Region &body, Value i, Value j, Value k) const {
    return isGemmNN(body, i, j, k);
  };

  LogicalResult matchAndRewriteNestedPattern(Operation *op,
                                             PatternRewriter &rewriter) const {
    AffineForOp loopI, loopJ, loopK = nullptr;
    auto body = [this, &loopI, &loopJ, &loopK](Operation &op) -> bool {
      // exit if this stage already fired.
      if (auto attr = op.getAttr("BLIS"))
        if (auto stringAttr = attr.dyn_cast<StringAttr>()) {
          auto step = std::stoi((stringAttr.getValue().str()));
          if (step >= 1)
            return false;
        }
      auto loop = cast<AffineForOp>(op);
      loopK = loop;
      auto k = loop.getInductionVar();
      auto parent = loop.getParentOfType<AffineForOp>();
      loopJ = parent;
      auto j = parent.getInductionVar();
      parent = parent.getParentOfType<AffineForOp>();
      loopI = parent;
      auto i = parent.getInductionVar();
      return succeeded(matchBody(loop.getLoopBody(), i, j, k));
    };
    {
      NestedPatternContext raii;
      using namespace matcher;
      auto m = For(For(For(body)));
      SmallVector<NestedMatch, 1> matches;
      m.match(op, &matches);
      if (matches.empty())
        return failure();
    }
    LLVM_DEBUG(llvm::dbgs() << "Matched Matmul pattern\n");
    SmallVector<AffineForOp, 3> loops = {loopJ, loopK, loopI};
    SmallVector<AffineForOp, 3> interchangedLoops = {};
    if (failed(interchangeBands(rewriter, loops, interchangedLoops, 1, 1)))
      return failure();
    return success();
  };
  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    return matchAndRewriteNestedPattern(op, rewriter);
  };
};

///
class BlisMatmulOpStepThree : public RewritePattern {
public:
  BlisMatmulOpStepThree(MLIRContext *context)
      : RewritePattern(AffineForOp::getOperationName(), 0, context) {}

  LogicalResult matchBody(Region &body, Value i, Value j, Value k) const {
    return isGemmNN(body, i, j, k);
  };

  LogicalResult matchAndRewriteNestedPattern(Operation *op,
                                             PatternRewriter &rewriter) const {
    AffineForOp loopI, loopJ, loopK, loopII, loopJJ, loopKK = nullptr;
    auto body = [this, &loopI, &loopJ, &loopK, &loopII, &loopJJ,
                 &loopKK](Operation &op) -> bool {
      if (alreadyFired(3, op))
        return false;
      auto loop = cast<AffineForOp>(op);
      loopII = loop;
      auto ii = loop.getInductionVar();
      auto parent = loop.getParentOfType<AffineForOp>();
      loopKK = parent;
      auto kk = parent.getInductionVar();
      parent = parent.getParentOfType<AffineForOp>();
      loopJJ = parent;
      auto jj = parent.getInductionVar();
      parent = parent.getParentOfType<AffineForOp>();
      loopI = parent;
      parent = parent.getParentOfType<AffineForOp>();
      loopK = parent;
      parent = parent.getParentOfType<AffineForOp>();
      loopJ = parent;
      return succeeded(matchBody(loop.getLoopBody(), ii, jj, kk));
    };
    {
      NestedPatternContext raii;
      using namespace matcher;
      auto m = For(For(For(For(For(For(body))))));
      SmallVector<NestedMatch, 1> matches;
      m.match(op, &matches);
      if (matches.empty())
        return failure();
    }
    LLVM_DEBUG(llvm::dbgs() << "Matched Matmul pattern\n");
    SmallVector<AffineForOp, 3> loops = {loopJ,  loopK,  loopI,
                                         loopJJ, loopII, loopKK};
    SmallVector<AffineForOp, 3> interchangedLoops = {};
    if (failed(interchangeBands(rewriter, loops, interchangedLoops, 3, 4)))
      return failure();
    rewriter.eraseOp(op);
    return success();
  };
  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    return matchAndRewriteNestedPattern(op, rewriter);
  };
};

///
class BlisMatmulOpStepFour : public RewritePattern {
public:
  BlisMatmulOpStepFour(MLIRContext *context)
      : RewritePattern(AffineForOp::getOperationName(), 0, context) {}

  LogicalResult matchBody(Region &body, Value i, Value j, Value k) const {
    return isGemmNN(body, i, j, k);
  };

  LogicalResult matchAndRewriteNestedPattern(Operation *op,
                                             PatternRewriter &rewriter) const {
    AffineForOp loopI, loopJ, loopK, loopII, loopJJ, loopKK = nullptr;
    auto body = [this, &loopI, &loopJ, &loopK, &loopII, &loopJJ,
                 &loopKK](Operation &op) -> bool {
      if (alreadyFired(4, op))
        return false;
      auto loop = cast<AffineForOp>(op);
      loopKK = loop;
      auto kk = loop.getInductionVar();
      auto parent = loop.getParentOfType<AffineForOp>();
      loopII = parent;
      auto ii = parent.getInductionVar();
      parent = parent.getParentOfType<AffineForOp>();
      loopJJ = parent;
      auto jj = parent.getInductionVar();
      parent = parent.getParentOfType<AffineForOp>();
      loopI = parent;
      parent = parent.getParentOfType<AffineForOp>();
      loopK = parent;
      parent = parent.getParentOfType<AffineForOp>();
      loopJ = parent;
      return succeeded(matchBody(loop.getLoopBody(), ii, jj, kk));
    };
    {
      NestedPatternContext raii;
      using namespace matcher;
      auto m = For(For(For(For(For(For(body))))));
      SmallVector<NestedMatch, 1> matches;
      m.match(op, &matches);
      if (matches.empty())
        return failure();
    }
    LLVM_DEBUG(llvm::dbgs() << "Matched Matmul pattern\n");
    SmallVector<AffineForOp, 6> loops = {loopJ,  loopK,  loopI,
                                         loopJJ, loopII, loopKK};
    SmallVector<unsigned, 6> tileSizes = {1, 1, 1, 4, 4, 1};
    if (failed(tilePerfectlyNestedBands(4, rewriter, loops, tileSizes)))
      return failure();
    return success();
  };
  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    return matchAndRewriteNestedPattern(op, rewriter);
  };
};

static void BlisPolyhedralOptImpl(FuncOp op, MLIRContext *context) {
  OwningRewritePatternList patterns;
  patterns.insert<BlisMatmulOpStepOne>(context);
  patterns.insert<BlisMatmulOpStepTwo>(context);
  patterns.insert<BlisMatmulOpStepThree>(context);
  // patterns.insert<BlisMatmulOpStepFour>(context);
  applyPatternsAndFoldGreedily(op, patterns);
}

namespace {

struct BlisPolyhedralOpt : public BlisPolyhedralOptBase<BlisPolyhedralOpt> {
  void runOnFunction() override {
    BlisPolyhedralOptImpl(getFunction(), &getContext());
  }
};

} // end namespace

std::unique_ptr<OperationPass<FuncOp>> mlir::createBlisPolyhedralOptPass() {
  return std::make_unique<BlisPolyhedralOpt>();
}
