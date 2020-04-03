#include "mlir/Conversion/RaiseAffineToLinalg/RaiseAffineToLinalgPass.h"
#include "mlir/Analysis/NestedMatcher.h"
#include "mlir/Conversion/RaiseAffineToLinalg/RaiseAffineToLinalg.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Linalg/EDSC/Intrinsics.h"
#include "mlir/Dialect/Linalg/IR/LinalgOps.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/Access.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/Support/Debug.h"

using namespace mlir;

namespace {
class RaiseAffineToLinalgPass : public FunctionPass<RaiseAffineToLinalgPass> {
  void runOnFunction() override;
};
} // namespace

#define DEBUG_TYPE "mlir-raise-to-linalg"

static llvm::cl::opt<bool>
    clEmitCall("emit-blas-call", llvm::cl::desc("directly emit blas calls."),
               llvm::cl::init(false));

namespace {

// insert a symbol reference to "fName", inserting it into the module
// if necessary.
static FlatSymbolRefAttr getOrInsertFunction(PatternRewriter &rewriter,
                                             ModuleOp module, std::string fName,
                                             const ArrayRef<Type> &types) {
  auto *context = module.getContext();
  if (module.lookupSymbol(fName))
    return SymbolRefAttr::get(fName, context);
  auto libFnInfoType = FunctionType::get(types, {}, rewriter.getContext());
  OpBuilder::InsertionGuard guard(rewriter);
  rewriter.setInsertionPoint(module.getBody(),
                             std::prev(module.getBody()->end()));
  rewriter.create<FuncOp>(module.getLoc(), fName, libFnInfoType,
                          ArrayRef<NamedAttribute>{});
  return SymbolRefAttr::get(fName, context);
}

// return a value representing the access into a global array with
// name "name", create the array if necessary.
static Value getOrCreateGlobalArray(Location loc, OpBuilder &builder,
                                    StringRef name,
                                    SmallVector<int64_t, 4> &values,
                                    ModuleOp module,
                                    LLVM::LLVMDialect *llvmDialect) {
  // create the global at the entry of the module.
  LLVM::GlobalOp global;
  if (!(global = module.lookupSymbol<LLVM::GlobalOp>(name))) {
    OpBuilder::InsertionGuard insertGuard(builder);
    builder.setInsertionPointToStart(module.getBody());
    auto type = LLVM::LLVMType::getArrayTy(
        LLVM::LLVMType::getInt8Ty(llvmDialect), values.size());
    auto attr = builder.getI64ArrayAttr(values);
    global =
        builder.create<LLVM::GlobalOp>(loc, type, true, LLVM::Linkage::Internal,
                                       name, builder.getArrayAttr(attr));
  }

  // Get the pointer to the first int in the global array.
  Value globalPtr = builder.create<LLVM::AddressOfOp>(loc, global);
  Value cst0 = builder.create<LLVM::ConstantOp>(
      loc, LLVM::LLVMType::getInt64Ty(llvmDialect),
      builder.getIntegerAttr(builder.getIndexType(), 0));
  return builder.create<LLVM::GEPOp>(loc,
                                     LLVM::LLVMType::getInt8PtrTy(llvmDialect),
                                     globalPtr, ArrayRef<Value>({cst0, cst0}));
}

static SmallVector<int64_t, 8> applyPermutation(ArrayRef<int64_t> shape,
                                                ArrayRef<int64_t> permutation) {
  assert((shape.size() == permutation.size()) && "must be equal");
  SmallVector<int64_t, 8> result{};
  for (size_t i = 0; i < shape.size(); i++) {
    result.push_back(shape[permutation[i]]);
  }
  return result;
}

// helper function for out-of-place transposition.
static MemRefType getTransposedMemref(MemRefType source,
                                      ArrayRef<int64_t> permutation, Type t) {
  auto sourceMemRefShape = source.getShape();
  auto res = MemRefType::get(applyPermutation(sourceMemRefShape, permutation),
                             t, {}, 0);
  return res;
}

/// Class for matching C[i][j][k] += A[i][l] * B[j][l][k]
class ContractionMatcherTranspose : public OpRewritePattern<AffineForOp> {
public:
  using OpRewritePattern<AffineForOp>::OpRewritePattern;

  LogicalResult matchBody(Region &body, Value i, Value j, Value k, Value l,
                          Value &operandA, Value &operandB,
                          Value &operandC) const {
    using namespace matchers;
    {
      AccessPatternContext pctx(body.getContext());

      auto _i = m_Placeholder();
      auto _j = m_Placeholder();
      auto _k = m_Placeholder();
      auto _l = m_Placeholder();

      auto _A = m_ArrayPlaceholder();
      auto _B = m_ArrayPlaceholder();
      auto _C = m_ArrayPlaceholder();

      auto a = m_Op<AffineLoadOp>(_A({_i, _l}));
      auto b = m_Op<AffineLoadOp>(_B({_j, _l, _k}));
      auto c = m_Op<AffineLoadOp>(_C({_i, _j, _k}));
      auto s = m_Op<AffineStoreOp>(_C({_i, _j, _k}));

      auto contractionOp = m_Op<AddFOp>(m_Op<MulFOp>(a, b), c);

      auto store = dyn_cast<AffineStoreOp>(*std::prev(body.front().end(), 2));
      if (!matchPattern(store, s))
        return failure();

      auto add =
          dyn_cast_or_null<AddFOp>(store.getValueToStore().getDefiningOp());
      if ((!add) || (!contractionOp.match(add)))
        return failure();

      if (std::distance(body.front().begin(), body.front().end()) != 7)
        return failure();

      if ((i != pctx[_i]) || (j != pctx[_j]) || (k != pctx[_k]) ||
          (l != pctx[_l]))
        return failure();

      operandA = pctx[_A];
      operandB = pctx[_B];
      operandC = pctx[_C];
    }
    return success();
  }

  LogicalResult transformOp(Operation *op, SmallVector<Value, 3> &operands,
                            PatternRewriter &rewriter) const {
    if (!clEmitCall) {
      // emit linalg.
      using namespace edsc;
      using namespace edsc::ops;
      using namespace edsc::intrinsics;

      auto permutationMap =
          AffineMap::getPermutationMap({1, 0, 2}, rewriter.getContext());
      auto transposedB = rewriter.create<linalg::TransposeOp>(
          op->getLoc(), operands[1], AffineMapAttr::get(permutationMap));
      ScopedContext scop(rewriter, op->getLoc());
      AffineExpr i, j, k;
      bindDims(op->getContext(), i, j, k);
      ValueHandle v(transposedB);
      auto reshapedB =
          linalg_reshape(v, ArrayRef<ArrayRef<AffineExpr>>{i, {j, k}});
      ValueHandle vv(operands[2]);
      auto reshapedC =
          linalg_reshape(vv, ArrayRef<ArrayRef<AffineExpr>>{i, {j, k}});

      SmallVector<Value, 3> newOperands;
      newOperands.push_back(operands[0]);
      newOperands.push_back(reshapedB);
      newOperands.push_back(reshapedC);
      linalg_generic_matmul(makeValueHandles(newOperands));
    } else {
      // emit blas.
      auto operandA = operands[0];
      auto operandC = operands[2];
      auto operandBType = operands[1].getType();
      auto module = op->getParentOfType<ModuleOp>();
      auto f32Type = FloatType::getF32(module.getContext());
      // auto *llvmDialect =
      //    op->getContext()->getRegisteredDialect<LLVM::LLVMDialect>();
      // create a global array representing the permutation.
      SmallVector<int64_t, 4> permutation = {1, 0, 2};
      // Value permutationVar =
      //    getOrCreateGlobalArray(op->getLoc(), rewriter, "permutation",
      //                           permutation, module, llvmDialect);
      // create an output buffer for the transposition.
      auto transposedBType = getTransposedMemref(
          operandBType.dyn_cast<MemRefType>(), permutation, f32Type);
      auto transposedB =
          rewriter.create<AllocOp>(op->getLoc(), transposedBType).getResult();
      // create transpose call.
      auto functionName = composeFunctionCallName(
          FUNCTION::TRANSPOSE, ArrayRef<Type>{operandBType, transposedBType});
      auto symbolFn =
          getOrInsertFunction(rewriter, module, functionName,
                              ArrayRef<Type>{operandBType, transposedBType});
      rewriter.setInsertionPoint(op);
      rewriter.create<CallOp>(op->getLoc(), symbolFn, ArrayRef<Type>{},
                              ArrayRef<Value>{operands[1], transposedB});

      // new buffer for reshaped tensors C and B.
      auto operandCType = operands[2].getType();
      auto operandCShape = operandCType.dyn_cast<MemRefType>().getShape();
      auto transposedBShape = transposedBType.getShape();
      auto reshapedCType = MemRefType::get(
          {operandCShape[0], operandCShape[1] * operandCShape[2]}, f32Type, {},
          0);
      auto reshapedBType = MemRefType::get(
          {transposedBShape[0], transposedBShape[1] * transposedBShape[2]},
          f32Type, {}, 0);
      auto reshapedC =
          rewriter.create<AllocOp>(op->getLoc(), reshapedCType).getResult();
      auto reshapedB =
          rewriter.create<AllocOp>(op->getLoc(), reshapedBType).getResult();

      // reshape B and C.
      functionName = composeFunctionCallName(
          FUNCTION::RESHAPE, ArrayRef<Type>{transposedBType, reshapedBType});
      symbolFn =
          getOrInsertFunction(rewriter, module, functionName,
                              ArrayRef<Type>{transposedBType, reshapedBType});
      rewriter.create<CallOp>(op->getLoc(), symbolFn, ArrayRef<Type>{},
                              ArrayRef<Value>{transposedB, reshapedB});
      functionName = composeFunctionCallName(
          FUNCTION::RESHAPE, ArrayRef<Type>{operandCType, reshapedCType});
      symbolFn =
          getOrInsertFunction(rewriter, module, functionName,
                              ArrayRef<Type>{operandCType, reshapedCType});
      rewriter.create<CallOp>(op->getLoc(), symbolFn, ArrayRef<Type>{},
                              ArrayRef<Value>{operandC, reshapedC});

      // matmul.
      auto operandAType = operandA.getType();
      functionName = composeFunctionCallName(
          FUNCTION::MATMUL,
          ArrayRef<Type>{operandAType, reshapedBType, reshapedCType});
      symbolFn = getOrInsertFunction(
          rewriter, module, functionName,
          ArrayRef<Type>{operandAType, reshapedBType, reshapedCType});
      rewriter.create<CallOp>(op->getLoc(), functionName, ArrayRef<Type>{},
                              ArrayRef<Value>{operandA, reshapedB, reshapedC});

      // reshaped C.
      functionName = composeFunctionCallName(
          FUNCTION::RESHAPE, ArrayRef<Type>{reshapedCType, operandCType});
      symbolFn =
          getOrInsertFunction(rewriter, module, functionName,
                              ArrayRef<Type>{reshapedCType, operandCType});
      rewriter.create<CallOp>(op->getLoc(), functionName, ArrayRef<Type>{},
                              ArrayRef<Value>{reshapedC, operandC});
    }
    return success();
  }

  LogicalResult matchAndRewriteNestedPattern(Operation *op,
                                             PatternRewriter &rewriter) const {
    Value operandA, operandB, operandC;
    auto body = [this, &operandA, &operandB, &operandC](Operation &op) -> bool {
      auto loop = cast<AffineForOp>(op);
      auto l = loop.getInductionVar();
      auto parent = loop.getParentOfType<AffineForOp>();
      auto k = parent.getInductionVar();
      parent = parent.getParentOfType<AffineForOp>();
      auto j = parent.getInductionVar();
      parent = parent.getParentOfType<AffineForOp>();
      auto i = parent.getInductionVar();
      return succeeded(matchBody(loop.getLoopBody(), i, j, k, l, operandA,
                                 operandB, operandC));
    };

    {
      NestedPatternContext raii;
      using namespace matcher;
      auto m = For(For(For(For(body))));
      SmallVector<NestedMatch, 1> matches;
      m.match(op, &matches);
      if (matches.empty())
        return failure();
    }

    if ((!operandA) || (!operandB) || (!operandC))
      return failure();

    LLVM_DEBUG(llvm::dbgs() << "Match Contraction with Transposed operand!\n");

    SmallVector<Value, 3> operands = {operandA, operandB, operandC};
    if (failed(transformOp(op, operands, rewriter)))
      return failure();

    rewriter.eraseOp(op);
    return success();
  }

  // main rewriting function.
  LogicalResult matchAndRewrite(AffineForOp op,
                                PatternRewriter &rewriter) const override {
    return matchAndRewriteNestedPattern(op, rewriter);
  }
};

/// Class for matching C[i][j][k] += A[i][l] * B[l][j][k].
class ContractionMatcher : public OpRewritePattern<AffineForOp> {
public:
  using OpRewritePattern<AffineForOp>::OpRewritePattern;

  LogicalResult matchBody(Region &body, Value i, Value j, Value k, Value l,
                          Value &operandA, Value &operandB,
                          Value &operandC) const {
    using namespace matchers;
    {
      AccessPatternContext pctx(body.getContext());

      auto _i = m_Placeholder();
      auto _j = m_Placeholder();
      auto _k = m_Placeholder();
      auto _l = m_Placeholder();

      auto _A = m_ArrayPlaceholder();
      auto _B = m_ArrayPlaceholder();
      auto _C = m_ArrayPlaceholder();

      auto a = m_Op<AffineLoadOp>(_A({_i, _l}));
      auto b = m_Op<AffineLoadOp>(_B({_l, _j, _k}));
      auto c = m_Op<AffineLoadOp>(_C({_i, _j, _k}));
      auto s = m_Op<AffineStoreOp>(_C({_i, _j, _k}));

      auto contractionOp = m_Op<AddFOp>(m_Op<MulFOp>(a, b), c);

      auto store = dyn_cast<AffineStoreOp>(*std::prev(body.front().end(), 2));
      if (!matchPattern(store, s))
        return failure();

      auto add =
          dyn_cast_or_null<AddFOp>(store.getValueToStore().getDefiningOp());
      if ((!add) || (!contractionOp.match(add)))
        return failure();

      if (std::distance(body.front().begin(), body.front().end()) != 7)
        return failure();

      if ((i != pctx[_i]) || (j != pctx[_j]) || (k != pctx[_k]) ||
          (l != pctx[_l]))
        return failure();

      operandA = pctx[_A];
      operandB = pctx[_B];
      operandC = pctx[_C];
    }
    return success();
  }

  LogicalResult matchAndRewriteNestedPattern(Operation *op,
                                             PatternRewriter &rewriter) const {
    Value operandA, operandB, operandC;
    auto body = [this, &operandA, &operandB, &operandC](Operation &op) -> bool {
      auto loop = cast<AffineForOp>(op);
      auto l = loop.getInductionVar();
      auto parent = loop.getParentOfType<AffineForOp>();
      auto k = parent.getInductionVar();
      parent = parent.getParentOfType<AffineForOp>();
      auto j = parent.getInductionVar();
      parent = parent.getParentOfType<AffineForOp>();
      auto i = parent.getInductionVar();
      return succeeded(matchBody(loop.getLoopBody(), i, j, k, l, operandA,
                                 operandB, operandC));
    };

    {
      NestedPatternContext raii;
      using namespace matcher;
      auto m = For(For(For(For(body))));
      SmallVector<NestedMatch, 1> matches;
      m.match(op, &matches);
      if (matches.empty())
        return failure();
    }

    if ((!operandA) || (!operandB) || (!operandC))
      return failure();

    LLVM_DEBUG(llvm::dbgs() << "Matched Contraction pattern!\n");

    SmallVector<Value, 3> operands = {operandA, operandB, operandC};
    if (clEmitCall) {
      // emit blas.
      auto module = op->getParentOfType<ModuleOp>();
      auto f32Type = FloatType::getF32(module.getContext());

      auto operandAType = operandA.getType();
      auto operandBType = operandB.getType();
      auto operandCType = operandC.getType();
      auto operandBShape = operandBType.dyn_cast<MemRefType>().getShape();
      auto operandCShape = operandCType.dyn_cast<MemRefType>().getShape();

      // new buffer for reshaped tensors C and B.
      auto reshapedCType = MemRefType::get(
          {operandCShape[0], operandCShape[1] * operandCShape[2]}, f32Type, {},
          0);
      auto reshapedBType = MemRefType::get(
          {operandBShape[0], operandBShape[1] * operandBShape[2]}, f32Type, {},
          0);
      auto reshapedC =
          rewriter.create<AllocOp>(op->getLoc(), reshapedCType).getResult();
      auto reshapedB =
          rewriter.create<AllocOp>(op->getLoc(), reshapedBType).getResult();

      // reshape B and C.
      auto functionName = composeFunctionCallName(
          FUNCTION::RESHAPE, ArrayRef<Type>{operandBType, reshapedBType});
      auto symbolFn =
          getOrInsertFunction(rewriter, module, functionName,
                              ArrayRef<Type>{operandBType, reshapedBType});
      rewriter.setInsertionPoint(op);
      rewriter.create<CallOp>(op->getLoc(), symbolFn, ArrayRef<Type>{},
                              ArrayRef<Value>{operands[1], reshapedB});

      symbolFn =
          getOrInsertFunction(rewriter, module, functionName,
                              ArrayRef<Type>{operandCType, reshapedCType});
      rewriter.create<CallOp>(op->getLoc(), symbolFn, ArrayRef<Type>{},
                              ArrayRef<Value>{operands[2], reshapedC});

      // matmul.
      functionName = composeFunctionCallName(
          FUNCTION::MATMUL,
          ArrayRef<Type>{operandAType, reshapedBType, reshapedCType});
      symbolFn = getOrInsertFunction(
          rewriter, module, functionName,
          ArrayRef<Type>{operandAType, reshapedBType, reshapedCType});
      rewriter.create<CallOp>(op->getLoc(), functionName, ArrayRef<Type>{},
                              ArrayRef<Value>{operandA, reshapedB, reshapedC});

      // reshape C.
      functionName = composeFunctionCallName(
          FUNCTION::RESHAPE, ArrayRef<Type>{reshapedCType, operandCType});
      symbolFn =
          getOrInsertFunction(rewriter, module, functionName,
                              ArrayRef<Type>{reshapedCType, operandCType});
      rewriter.create<CallOp>(op->getLoc(), functionName, ArrayRef<Type>{},
                              ArrayRef<Value>{reshapedC, operandC});

    } else {
      // emit linalg.
      using namespace edsc;
      using namespace edsc::ops;
      using namespace edsc::intrinsics;

      ScopedContext scop(rewriter, op->getLoc());
      AffineExpr i, j, k;
      bindDims(op->getContext(), i, j, k);
      ValueHandle v(operands[1]); // reshape B
      auto reshapedB =
          linalg_reshape(v, ArrayRef<ArrayRef<AffineExpr>>{i, {j, k}});
      ValueHandle vv(operands[2]); // reshape C
      auto reshapedC =
          linalg_reshape(vv, ArrayRef<ArrayRef<AffineExpr>>{i, {j, k}});

      SmallVector<Value, 4> newOperands;
      newOperands.push_back(operands[0]);
      newOperands.push_back(reshapedB);
      newOperands.push_back(reshapedC);
      linalg_generic_matmul(makeValueHandles(newOperands)); // mul A * B += C
    }
    rewriter.eraseOp(op);
    return success();
  }

  // main rewriting function.
  LogicalResult matchAndRewrite(AffineForOp op,
                                PatternRewriter &rewriter) const override {
    return matchAndRewriteNestedPattern(op, rewriter);
  }
};

/// Class for matching a matmul op.
class MatMulMatcher : public OpRewritePattern<AffineForOp> {
public:
  using OpRewritePattern<AffineForOp>::OpRewritePattern;

  LogicalResult matchBody(Region &body, Value i, Value j, Value k,
                          Value &operandA, Value &operandB,
                          Value &operandC) const {
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

      // TODO: relax the order of operands and introduce
      // ArrayPlaceholder.
      auto matMulOp = m_Op<AddFOp>(c, m_Op<MulFOp>(a, b));

      auto store = dyn_cast<AffineStoreOp>(*std::prev(body.front().end(), 2));

      if (!matchPattern(store, m_Op<AffineStoreOp>(_C({_i, _j}))))
        return failure();

      auto add =
          dyn_cast_or_null<AddFOp>(store.getValueToStore().getDefiningOp());
      if ((!add) || (!matMulOp.match(add)))
        return failure();

      // TODO: we may be lenient to operations without side-effects, but they
      // should have been removed by DCE beforehand.
      if (std::distance(body.front().begin(), body.front().end()) != 7)
        return failure();

      if ((i != pctx[_i]) || (j != pctx[_j]) || (k != pctx[_k]))
        return failure();

      operandA = pctx[_A];
      operandB = pctx[_B];
      operandC = pctx[_C];
    }
    return success();
  }

  LogicalResult matchAndRewriteNestedPattern(Operation *op,
                                             PatternRewriter &rewriter) const {
    Value operandA, operandB, operandC;
    auto body = [this, &operandA, &operandB, &operandC](Operation &op) -> bool {
      auto loop = cast<AffineForOp>(op);
      Value k = loop.getInductionVar();
      auto parent = loop.getParentOfType<AffineForOp>();
      Value j = parent.getInductionVar();
      parent = parent.getParentOfType<AffineForOp>();
      Value i = parent.getInductionVar();
      return succeeded(
          matchBody(loop.getLoopBody(), i, j, k, operandA, operandB, operandC));
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

    if ((!operandA) || (!operandB) || (!operandC))
      return failure();

    LLVM_DEBUG(llvm::dbgs() << "Matched Matmul pattern!\n");

    SmallVector<Value, 3> operands = {operandA, operandB, operandC};
    if (clEmitCall) {
      // emit blas call.
      auto module = op->getParentOfType<ModuleOp>();
      auto operandAType = operandA.getType();
      auto operandBType = operandB.getType();
      auto operandCType = operandC.getType();
      auto functionName = composeFunctionCallName(
          FUNCTION::MATMUL,
          ArrayRef<Type>{operandAType, operandBType, operandCType});
      auto symbolFn = getOrInsertFunction(
          rewriter, module, functionName,
          ArrayRef<Type>{operandAType, operandBType, operandCType});
      rewriter.setInsertionPoint(op);
      rewriter.create<CallOp>(op->getLoc(), symbolFn, ArrayRef<Type>{},
                              operands);
    } else {
      // emit linalg.
      using namespace edsc;
      using namespace edsc::ops;
      ScopedContext scop(rewriter, op->getLoc());
      linalg_generic_matmul(makeValueHandles(operands));
    }
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

void RaiseAffineToLinalgPass::runOnFunction() {

  ConversionTarget target(getContext());
  target.addLegalDialect<mlir::linalg::LinalgDialect, StandardOpsDialect,
                         LLVM::LLVMDialect>();
  target.addLegalOp<ReturnOp, AllocOp, ModuleOp, ModuleTerminatorOp, FuncOp>();

  OwningRewritePatternList patterns;

  patterns.insert<MatMulMatcher>(&getContext());
  patterns.insert<ContractionMatcher>(&getContext());
  patterns.insert<ContractionMatcherTranspose>(&getContext());

  // run full conversion. As we discussed we want only
  // library calls.
  auto function = getFunction();
  if (failed(applyFullConversion(function, target, patterns)))
    signalPassFailure();
}

std::unique_ptr<OpPassBase<FuncOp>> mlir::createRaiseAffineToLinalgPass() {
  return std::make_unique<RaiseAffineToLinalgPass>();
}
