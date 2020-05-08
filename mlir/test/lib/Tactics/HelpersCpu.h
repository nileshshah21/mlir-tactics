#ifndef HELPERS_GENERATION_TACTICS_CPU
#define HELPERS_GENERATION_TACTICS_CPU

#include "HelpersLinalg.h"

/*
  Helpers for tablegen code generation - cpu path.
*/

namespace {

enum class FUNCTION {
  MATMUL,
  RESHAPE,
  TRANSPOSE,
  MATVEC,
};

std::string composeFunctionNameForTranspose(llvm::ArrayRef<mlir::Type> types) {
  std::string result = "transpose_";
  auto SShape = types[0].dyn_cast<mlir::MemRefType>().getShape();
  auto DShape = types[1].dyn_cast<mlir::MemRefType>().getShape();
  for (size_t i = 0; i < SShape.size() - 1; i++)
    result += std::to_string(SShape[i]) + "x";
  result += std::to_string(SShape[SShape.size() - 1]);
  result += "_to_";
  for (size_t i = 0; i < DShape.size() - 1; i++)
    result += std::to_string(DShape[i]) + "x";
  result += std::to_string(DShape[DShape.size() - 1]);
  return result;
}

llvm::SmallVector<int64_t, 8>
applyPermutation(llvm::ArrayRef<int64_t> shape,
                 llvm::ArrayRef<int64_t> permutation) {
  assert((shape.size() == permutation.size()) && "must be equal");
  llvm::SmallVector<int64_t, 8> result{};
  for (size_t i = 0; i < shape.size(); i++) {
    result.push_back(shape[permutation[i]]);
  }
  return result;
}

mlir::MemRefType getTransposedMemref(mlir::MemRefType source,
                                     llvm::ArrayRef<int64_t> permutation) {
  auto sourceMemRefShape = source.getShape();
  auto res =
      mlir::MemRefType::get(applyPermutation(sourceMemRefShape, permutation),
                            source.getElementType(), {}, 0);
  return res;
}

/// Detect whether memref dims [dim, dim + extent) can be reshaped without
/// copies.
bool isReshapableDimBand(unsigned dim, unsigned extent,
                         llvm::ArrayRef<int64_t> sizes,
                         llvm::ArrayRef<mlir::AffineExpr> strides) {
  assert(sizes.size() == strides.size() && "mismatched ranks");
  // off by 1 indexing to avoid out of bounds
  //                       V
  for (auto idx = dim, e = dim + extent; idx + 1 < e; ++idx) {
    // Only bands of static shapes are reshapable. This is due to the fact that
    // there is no relation between dynamic sizes and dynamic strides: we do not
    // have enough information to know whether a "-1" size corresponds to the
    // proper symbol in the AffineExpr of a stride.
    if (mlir::ShapedType::isDynamic(sizes[dim + 1]))
      return false;
    // TODO(ntv) Refine this by passing the proper nDims and nSymbols so we can
    // simplify on the fly and catch more reshapable cases.
    if (strides[idx] != strides[idx + 1] * sizes[idx + 1])
      return false;
  }
  return true;
}

bool isReassociationValid(llvm::ArrayRef<mlir::AffineMap> reassociation,
                          int *invalidIndex = nullptr) {
  if (reassociation.empty())
    return true;
  unsigned nDims = reassociation[0].getNumDims();
  unsigned nextExpectedDim = 0;
  for (auto it : llvm::enumerate(reassociation)) {
    auto m = it.value();
    if (m.getNumDims() != nDims || m.getNumSymbols() != 0) {
      if (invalidIndex)
        *invalidIndex = it.index();
      return false;
    }
    for (auto e : m.getResults()) {
      auto d = e.dyn_cast<mlir::AffineDimExpr>();
      if (!d || d.getPosition() != nextExpectedDim++) {
        if (invalidIndex)
          *invalidIndex = it.index();
        return false;
      }
    }
  }
  if (nextExpectedDim != nDims) {
    if (invalidIndex)
      *invalidIndex = reassociation.size() - 1;
    return false;
  }
  return true;
}

/// Compute the MemRefType obtained by applying the `reassociation` (which is
/// expected to be valid) to `type`.
/// If `type` is Contiguous MemRefType, this always produce a contiguous
/// MemRefType.
mlir::MemRefType
computeReshapeCollapsedType(mlir::MemRefType type,
                            llvm::ArrayRef<mlir::AffineMap> reassociation) {
  auto sizes = type.getShape();
  mlir::AffineExpr offset;
  llvm::SmallVector<mlir::AffineExpr, 4> strides;
  auto status = getStridesAndOffset(type, strides, offset);
  (void)status;
  assert(succeeded(status) && "expected strided memref");

  llvm::SmallVector<int64_t, 4> newSizes;
  newSizes.reserve(reassociation.size());
  llvm::SmallVector<mlir::AffineExpr, 4> newStrides;
  newStrides.reserve(reassociation.size());

  // Use the fact that reassociation is valid to simplify the logic: only use
  // each map's rank.
  assert(isReassociationValid(reassociation) && "invalid reassociation");
  unsigned currentDim = 0;
  for (mlir::AffineMap m : reassociation) {
    unsigned dim = m.getNumResults();
    int64_t size = 1;
    mlir::AffineExpr stride = strides[currentDim + dim - 1];
    if (!isReshapableDimBand(currentDim, dim, sizes, strides)) {
      size = mlir::ShapedType::kDynamicSize;
      stride = mlir::AffineExpr();
    } else {
      for (unsigned d = 0; d < dim; ++d)
        size *= sizes[currentDim + d];
    }
    newSizes.push_back(size);
    newStrides.push_back(stride);
    currentDim += dim;
  }

  // Early-exit: if `type` is contiguous, the result must be contiguous.
  if (canonicalizeStridedLayout(type).getAffineMaps().empty())
    return mlir::MemRefType::Builder(type).setShape(newSizes).setAffineMaps({});

  // Convert back to int64_t because we don't have enough information to create
  // new strided layouts from AffineExpr only. This corresponds to a case where
  // copies may be necessary.
  int64_t intOffset = mlir::ShapedType::kDynamicStrideOrOffset;
  if (auto o = offset.dyn_cast<mlir::AffineConstantExpr>())
    intOffset = o.getValue();
  llvm::SmallVector<int64_t, 4> intStrides;
  intStrides.reserve(strides.size());
  for (auto stride : newStrides) {
    if (auto cst = stride.dyn_cast_or_null<mlir::AffineConstantExpr>())
      intStrides.push_back(cst.getValue());
    else
      intStrides.push_back(mlir::ShapedType::kDynamicStrideOrOffset);
  }
  auto layout =
      makeStridedLinearLayoutMap(intStrides, intOffset, type.getContext());
  return canonicalizeStridedLayout(
      mlir::MemRefType::Builder(type).setShape(newSizes).setAffineMaps(
          {layout}));
}

template <typename AffineExprTy>
unsigned
getMaxPosOfType(llvm::ArrayRef<llvm::ArrayRef<mlir::AffineExpr>> exprArrays) {
  unsigned pos = 0;
  for (auto exprs : exprArrays) {
    for (auto expr : exprs) {
      expr.walk([&pos](mlir::AffineExpr e) {
        if (auto d = e.dyn_cast<AffineExprTy>())
          pos = std::max(pos, d.getPosition());
      });
    }
  }
  return pos;
}

llvm::SmallVector<mlir::AffineMap, 4> getSymbolLessAffineMaps(
    llvm::ArrayRef<llvm::ArrayRef<mlir::AffineExpr>> reassociation) {
  unsigned maxDim = getMaxPosOfType<mlir::AffineDimExpr>(reassociation);
  assert(getMaxPosOfType<mlir::AffineSymbolExpr>(reassociation) == 0 &&
         "Expected symbol-less expressions");
  llvm::SmallVector<mlir::AffineMap, 4> maps;
  maps.reserve(reassociation.size());
  for (auto exprs : reassociation) {
    assert(exprs.size() != 0);
    maps.push_back(
        mlir::AffineMap::get(maxDim + 1, 0, exprs, exprs[0].getContext()));
  }
  return maps;
}

mlir::MemRefType
getReshapedMemRef(mlir::MemRefType source,
                  llvm::ArrayRef<llvm::ArrayRef<int64_t>> reshapeMap) {
  assert(reshapeMap.size() == 2 && "expect two partition");
  auto indexPartitionOne = reshapeMap[0];
  auto indexPartitionTwo = reshapeMap[1];
  assert(indexPartitionOne.size() && "must be non empty");
  assert(indexPartitionTwo.size() && "must be non empty");
  llvm::SmallVector<mlir::AffineExpr, 4> dimPartitionOne;
  llvm::SmallVector<mlir::AffineExpr, 4> dimPartitionTwo;

  auto ctx = source.getContext();
  // create affine exprs using the position
  // specified in the 'indexPartitionOne' and 'indexPartitionTwo'
  // arrays.
  size_t size = indexPartitionOne.size();
  for (size_t i = 0; i < size; i++) {
    mlir::AffineExpr expr;
    bindDims(ctx, expr, static_cast<int>(indexPartitionOne[i]));
    dimPartitionOne.push_back(expr);
  }
  size = indexPartitionTwo.size();
  for (size_t i = 0; i < size; i++) {
    mlir::AffineExpr expr;
    bindDims(ctx, expr, static_cast<int>(indexPartitionTwo[i]));
    dimPartitionTwo.push_back(expr);
  }
  llvm::SmallVector<mlir::AffineMap, 4> maps;
  if (std::find(indexPartitionOne.begin(), indexPartitionOne.end(), 0) !=
      indexPartitionOne.end())
    maps = getSymbolLessAffineMaps({dimPartitionOne, {dimPartitionTwo}});
  else
    maps = getSymbolLessAffineMaps({{dimPartitionTwo}, dimPartitionOne});

  return computeReshapeCollapsedType(source, maps);
}

std::string composeFunctionNameForMatmul(llvm::ArrayRef<mlir::Type> types) {
  assert((types.size() == 3) && "expect 3 types");
  auto AShape = types[1].dyn_cast<mlir::MemRefType>().getShape();
  auto CShape = types[0].dyn_cast<mlir::MemRefType>().getShape();
  std::string result = "matmul_";
  result += std::to_string(CShape[0]) + "x" + std::to_string(CShape[1]) + "x" +
            std::to_string(AShape[1]);
  return result;
}

std::string composeFunctionNameForReshape(llvm::ArrayRef<mlir::Type> types) {
  if (types.size() != 2)
    llvm_unreachable("expect single memref");
  std::string result = "reshape_";
  auto SShape = types[0].dyn_cast<mlir::MemRefType>().getShape();
  auto DShape = types[1].dyn_cast<mlir::MemRefType>().getShape();
  for (size_t i = 0; i < SShape.size() - 1; i++)
    result += std::to_string(SShape[i]) + "x";
  result += std::to_string(SShape[SShape.size() - 1]);
  result += "_to_";
  for (size_t i = 0; i < DShape.size() - 1; i++)
    result += std::to_string(DShape[i]) + "x";
  result += std::to_string(DShape[DShape.size() - 1]);
  return result;
}

std::string composeFunctionNameForMatvec(llvm::ArrayRef<mlir::Type> types) {
  assert((types.size() == 3) && "expect 3 types");
  auto AShape = (types[0].dyn_cast<mlir::MemRefType>().getShape().size() == 2)
                    ? types[0].dyn_cast<mlir::MemRefType>().getShape()
                    : types[1].dyn_cast<mlir::MemRefType>().getShape();
  assert((AShape.size() == 2) && "expect 2-d array");
  auto xShape = types[2].dyn_cast<mlir::MemRefType>().getShape();
  assert((xShape.size() == 1) && "expect 1-d array");
  std::string result = "matvec_";
  result += std::to_string(AShape[0]) + "x" + std::to_string(AShape[1]) + "x" +
            std::to_string(xShape[0]);
  return result;
}

template <typename... Args>
std::string composeFunctionCallName(FUNCTION id, const Args... args) {
  llvm::ArrayRef<mlir::Type> types = {args...};
  switch (id) {
  case FUNCTION::MATMUL:
    return composeFunctionNameForMatmul(types);
  case FUNCTION::RESHAPE:
    return composeFunctionNameForReshape(types);
  case FUNCTION::TRANSPOSE:
    return composeFunctionNameForTranspose(types);
  case FUNCTION::MATVEC:
    return composeFunctionNameForMatvec(types);
  }
  assert(0 && "case not convered");
  return "nullptr";
}

// return a constant op. The value of the constant is the size of
// the permutation array.
mlir::Value
getPermutationSizeAsConstantOp(mlir::Location loc, mlir::OpBuilder &builder,
                               llvm::ArrayRef<int> permutation,
                               mlir::LLVM::LLVMDialect *llvmDialect) {
  auto llvmInt32Type = mlir::LLVM::LLVMType::getInt32Ty(llvmDialect);
  mlir::Value size = builder.create<mlir::LLVM::ConstantOp>(
      loc, llvmInt32Type, builder.getI32IntegerAttr(permutation.size()));
  return size;
}

// return a unique name for the permuation array.
std::string getPermutationArrayName(llvm::ArrayRef<int> perm) {
  std::string res = "permutation_";
  for (size_t i = 0; i < perm.size() - 1; i++)
    res += std::to_string(perm[i]) + "x";
  res += std::to_string(perm[perm.size() - 1]);
  return res;
}

mlir::Value createConstantFloatOp(int constant, mlir::Type t,
                                  mlir::PatternRewriter &rewriter,
                                  mlir::Location &loc) {
  return rewriter.create<mlir::ConstantOp>(loc, t,
                                           rewriter.getFloatAttr(t, constant));
}

// TODO: check how we can remove this function.
mlir::Value createConstantFloatOp(mlir::Value constant, mlir::Type t,
                                  mlir::PatternRewriter &rewriter,
                                  mlir::Location &loc) {
  return constant;
}

// create a function call to mkl sgemm. Declare the function if not already
// in the module.
// The generated function has the following args:
//
// llvm.i32 = transA
// llvm.i32 = transB
// memref<...> = C
// memref<...> = A
// memref<...> = B
// type of C alpha
// type of C beta
template <typename TypeAlpha, typename TypeBeta>
void createCallToMklSgemm(mlir::ModuleOp module,
                          mlir::PatternRewriter &rewriter, mlir::Location loc,
                          mlir::Value C, mlir::Value A, mlir::Value B,
                          TypeAlpha alpha, TypeBeta beta, int transA,
                          int transB, int64_t dimForM, int64_t dimForN,
                          int64_t dimForK) {
  static_assert(((std::is_same<TypeBeta, mlir::Value>::value) ||
                 (std::is_same<TypeBeta, int>::value)),
                "expect mlir::Value or int");
  static_assert(((std::is_same<TypeAlpha, mlir::Value>::value) ||
                 (std::is_same<TypeAlpha, int>::value)),
                "expect mlir::Value or int");
  // create alpha and beta using the same
  // type as the C.
  auto memref = C.getType().dyn_cast<mlir::MemRefType>();
  auto type = memref.getElementType();
  // create alpha and beta if specified as string.
  auto betaV = createConstantFloatOp(beta, type, rewriter, loc);
  auto alphaV = createConstantFloatOp(alpha, type, rewriter, loc);

  auto i64Type = mlir::LLVM::LLVMType::getInt64Ty(getLLVMDialect(module));
  auto i32Type = mlir::LLVM::LLVMType::getInt32Ty(getLLVMDialect(module));
  mlir::Value dimForMV = rewriter.create<mlir::LLVM::ConstantOp>(
      loc, i64Type, rewriter.getI64IntegerAttr(dimForM));
  mlir::Value dimForNV = rewriter.create<mlir::LLVM::ConstantOp>(
      loc, i64Type, rewriter.getI64IntegerAttr(dimForN));
  mlir::Value dimForKV = rewriter.create<mlir::LLVM::ConstantOp>(
      loc, i64Type, rewriter.getI64IntegerAttr(dimForK));
  // create TransA and TransB (as int).
  mlir::Value transAV = rewriter.create<mlir::LLVM::ConstantOp>(
      loc, i32Type, rewriter.getI32IntegerAttr(transA));
  mlir::Value transBV = rewriter.create<mlir::LLVM::ConstantOp>(
      loc, i32Type, rewriter.getI32IntegerAttr(transB));

  auto fn = composeFunctionCallName(
      FUNCTION::MATMUL,
      llvm::ArrayRef<mlir::Type>{C.getType(), A.getType(), B.getType()});
  auto symbolFn = getOrInsertFunction(
      rewriter, module, fn,
      llvm::ArrayRef<mlir::Type>{i32Type, i32Type, C.getType(), A.getType(),
                                 B.getType(), type, type, i64Type, i64Type,
                                 i64Type});
  rewriter.create<mlir::CallOp>(
      loc, symbolFn, llvm::ArrayRef<mlir::Type>{},
      llvm::ArrayRef<mlir::Value>{transAV, transBV, C, A, B, alphaV, betaV,
                                  dimForMV, dimForNV, dimForKV});
}

// create a function call to mkl sgemv. Declare the function if not already
// in the module.
template <typename TypeAlpha, typename TypeBeta>
void createCallToMklSgemv(mlir::ModuleOp module,
                          mlir::PatternRewriter &rewriter, mlir::Location loc,
                          mlir::Value x, mlir::Value A, mlir::Value y,
                          TypeAlpha alpha, TypeBeta beta, int transA) {
  static_assert(((std::is_same<TypeBeta, mlir::Value>::value) ||
                 (std::is_same<TypeBeta, int>::value)),
                "expect mlir::Value or int");
  static_assert(((std::is_same<TypeAlpha, mlir::Value>::value) ||
                 (std::is_same<TypeAlpha, int>::value)),
                "expect mlir::Value or int");
  // create alpha and beta using the same
  // type as the x, if specified as string.
  auto memref = x.getType().dyn_cast<mlir::MemRefType>();
  auto type = memref.getElementType();

  auto betaV = createConstantFloatOp(beta, type, rewriter, loc);
  auto alphaV = createConstantFloatOp(alpha, type, rewriter, loc);
  auto i32Type = mlir::LLVM::LLVMType::getInt32Ty(getLLVMDialect(module));
  mlir::Value transAV = rewriter.create<mlir::LLVM::ConstantOp>(
      loc, i32Type, rewriter.getI32IntegerAttr(transA));

  auto fn = composeFunctionCallName(
      FUNCTION::MATVEC,
      llvm::ArrayRef<mlir::Type>{x.getType(), A.getType(), y.getType()});
  auto symbolFn = getOrInsertFunction(
      rewriter, module, fn,
      llvm::ArrayRef<mlir::Type>{x.getType(), A.getType(), y.getType(), type,
                                 type, i32Type});
  rewriter.create<mlir::CallOp>(
      loc, symbolFn, llvm::ArrayRef<mlir::Type>{},
      llvm::ArrayRef<mlir::Value>{x, A, y, alphaV, betaV, transAV});
}

// create a function call to mkl rehsape. Declare the function if not already
// in the module.
void createCallToMklReshape(mlir::ModuleOp module,
                            mlir::PatternRewriter &rewriter, mlir::Location loc,
                            mlir::Value source, mlir::Value dest) {
  auto fn = composeFunctionCallName(
      FUNCTION::RESHAPE,
      llvm::ArrayRef<mlir::Type>{source.getType(), dest.getType()});
  auto symbolFn = getOrInsertFunction(
      rewriter, module, fn,
      llvm::ArrayRef<mlir::Type>{source.getType(), dest.getType()});
  rewriter.create<mlir::CallOp>(loc, symbolFn, llvm::ArrayRef<mlir::Type>{},
                                llvm::ArrayRef<mlir::Value>{source, dest});
}

// create a function call to mkl transpose. Declare the function if not already
// in the module. Create a global array containing the permuation indexes.
void createCallToMklTranspose(mlir::ModuleOp module,
                              mlir::PatternRewriter &rewriter,
                              mlir::Location loc, mlir::Value source,
                              mlir::Value dest, llvm::ArrayRef<int> perm) {
  auto fn = composeFunctionCallName(
      FUNCTION::TRANSPOSE,
      llvm::ArrayRef<mlir::Type>{source.getType(), dest.getType()});
  auto *llvmDialect = getLLVMDialect(module);
  auto global = getOrCreateGlobalArray(
      loc, rewriter, getPermutationArrayName(perm), perm, module, llvmDialect);
  auto size = getPermutationSizeAsConstantOp(loc, rewriter, perm, llvmDialect);
  auto symbolFn = getOrInsertFunction(
      rewriter, module, fn,
      llvm::ArrayRef<mlir::Type>{source.getType(), dest.getType(),
                                 global.getType(), size.getType()});
  rewriter.create<mlir::CallOp>(
      loc, symbolFn, llvm::ArrayRef<mlir::Type>{},
      llvm::ArrayRef<mlir::Value>{source, dest, global, size});
}

} // end namespace

#endif
