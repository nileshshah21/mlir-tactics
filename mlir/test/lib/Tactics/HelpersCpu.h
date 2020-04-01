#ifndef HELPERS_GENERATION_TACTICS_CPU
#define HELPERS_GENERATION_TACTICS_CPU

#include "HelpersCommon.h"

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

std::string
composeFunctionNameForTranspose(const llvm::ArrayRef<mlir::Type> &types) {
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

// check that the elements in the vector are consecutive
// integer.
// {1, 2} -> ok
// {1, 1} not ok. (we cannot use std::is_sorted)
bool areConsecutive(llvm::ArrayRef<int64_t> indexMap) {
  bool isTrue = true;
  for (size_t i = 1; i < indexMap.size(); i++) {
    if (indexMap[i] != indexMap[i - 1] + 1) {
      isTrue = false;
      break;
    }
  }
  return isTrue;
}

// get dimensions not involved in the reshape operation.
llvm::SmallVector<int64_t, 8>
getOtherDimensions(size_t numberOfDims, llvm::ArrayRef<int64_t> indexMap) {
  assert(areConsecutive(indexMap) && "expect consecutive dimensions");
  llvm::SmallVector<int64_t, 8> allIndexes;
  // create array containing all indexes for each dimension.
  // (i.e., for numberOfDims = 3 -> {0, 1, 2}
  int64_t dim = 0;
  for (size_t i = 0; i < numberOfDims; i++)
    allIndexes.push_back(dim++);
  // subtract to allIndexes
  // the indexes of the dimensions to be reshaped.
  llvm::SmallVector<int64_t, 8> difference{};
  std::set_difference(allIndexes.begin(), allIndexes.end(), indexMap.begin(),
                      indexMap.end(), std::back_inserter(difference));
  return difference;
}

llvm::SmallVector<int64_t, 8> applyIndexMap(llvm::ArrayRef<int64_t> shape,
                                            llvm::ArrayRef<int64_t> indexMap) {
  assert((shape.size() > indexMap.size()) && "shape must be > than indexMap");

  llvm::SmallVector<int64_t, 8> result{};
  assert((areConsecutive(indexMap)) && "expect consecutive elements");
  int64_t newDim = 1;
  for (size_t i = 0; i < indexMap.size(); i++) {
    newDim *= shape[indexMap[i]];
  }
  auto dimensionNotInIndexMap = getOtherDimensions(shape.size(), indexMap);

  assert((dimensionNotInIndexMap.size() >= 1) && "expect at least one element");
  assert((indexMap.size() >= 1) && "expect at least one element");

  if (dimensionNotInIndexMap[0] < indexMap[0]) {
    for (const auto dim : dimensionNotInIndexMap)
      result.push_back(shape[dim]);
    result.push_back(newDim);
    return result;
  }

  if (dimensionNotInIndexMap[0] > indexMap[0]) {
    result.push_back(newDim);
    for (const auto dim : dimensionNotInIndexMap)
      result.push_back(shape[dim]);
    return result;
  }
  assert(0 && "dimensionNotInIndexMap and indexMap cannot have same element");
  return result;
}

mlir::MemRefType getReshapedMemRef(mlir::MemRefType source,
                                   llvm::ArrayRef<int64_t> indexMap) {
  auto sourceMemRefShape = source.getShape();
  auto res = mlir::MemRefType::get(applyIndexMap(sourceMemRefShape, indexMap),
                                   source.getElementType(), {}, 0);
  return res;
}

std::string
composeFunctionNameForMatmul(const llvm::ArrayRef<mlir::Type> &types) {
  assert((types.size() == 3) && "expect 3 types");
  auto AShape = types[1].dyn_cast<mlir::MemRefType>().getShape();
  auto CShape = types[0].dyn_cast<mlir::MemRefType>().getShape();
  std::string result = "matmul_";
  result += std::to_string(CShape[0]) + "x" + std::to_string(CShape[1]) + "x" +
            std::to_string(AShape[1]);
  return result;
}

std::string
composeFunctionNameForReshape(const llvm::ArrayRef<mlir::Type> &types) {
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

std::string
composeFunctionNameForMatvec(const llvm::ArrayRef<mlir::Type> &types) {
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
                               const llvm::ArrayRef<int> &permutation,
                               mlir::LLVM::LLVMDialect *llvmDialect) {
  auto llvmInt32Type = mlir::LLVM::LLVMType::getInt32Ty(llvmDialect);
  mlir::Value size = builder.create<mlir::LLVM::ConstantOp>(
      loc, llvmInt32Type, builder.getI32IntegerAttr(permutation.size()));
  return size;
}

// return a unique name for the permuation array.
std::string getPermutationArrayName(const llvm::ArrayRef<int> &perm) {
  std::string res = "permutation_";
  for (size_t i = 0; i < perm.size() - 1; i++)
    res += std::to_string(perm[i]) + "x";
  res += std::to_string(perm[perm.size() - 1]);
  return res;
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
// llvm.i64 alpha
// llvm.i64 beta
void createCallToMklSgemm(mlir::ModuleOp module,
                          mlir::PatternRewriter &rewriter, mlir::Location loc,
                          mlir::Value C, mlir::Value A, mlir::Value B,
                          int64_t alpha, int64_t beta, int transA, int transB,
                          int64_t dimForM, int64_t dimForN, int64_t dimForK) {
  auto i64Type = mlir::LLVM::LLVMType::getInt64Ty(getLLVMDialect(module));
  auto i32Type = mlir::LLVM::LLVMType::getInt32Ty(getLLVMDialect(module));
  // create alpha and beta.
  mlir::Value alphaV = rewriter.create<mlir::LLVM::ConstantOp>(
      loc, i64Type, rewriter.getI64IntegerAttr(alpha));
  mlir::Value betaV = rewriter.create<mlir::LLVM::ConstantOp>(
      loc, i64Type, rewriter.getI64IntegerAttr(beta));
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
                                 B.getType(), i64Type, i64Type, i64Type,
                                 i64Type, i64Type});
  rewriter.create<mlir::CallOp>(
      loc, symbolFn, llvm::ArrayRef<mlir::Type>{},
      llvm::ArrayRef<mlir::Value>{transAV, transBV, C, A, B, alphaV, betaV,
                                  dimForMV, dimForNV, dimForKV});
}

// create a function call to mkl sgemv. Declare the function if not already
// in the module.
void createCallToMklSgemv(mlir::ModuleOp module,
                          mlir::PatternRewriter &rewriter, mlir::Location loc,
                          mlir::Value x, mlir::Value A, mlir::Value y) {
  auto fn = composeFunctionCallName(
      FUNCTION::MATVEC,
      llvm::ArrayRef<mlir::Type>{x.getType(), A.getType(), y.getType()});
  auto symbolFn = getOrInsertFunction(
      rewriter, module, fn,
      llvm::ArrayRef<mlir::Type>{x.getType(), A.getType(), y.getType()});
  rewriter.create<mlir::CallOp>(loc, symbolFn, llvm::ArrayRef<mlir::Type>{},
                                llvm::ArrayRef<mlir::Value>{x, A, y});
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
