#ifndef HELPERS_GENERATION_TACTICS
#define HELPERS_GENERATION_TACTICS

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
#include <iostream>
#include <string>

/*
  Helpers for tablegen code generation
*/

namespace {

enum class FUNCTION {
  MATMUL,
  RESHAPE,
  TRANSPOSE,
};

std::string
composeFunctionNameForTranspose(const llvm::ArrayRef<mlir::Type> &types) {
  std::string result = "Transpose_";
  auto TShape = types[0].dyn_cast<mlir::MemRefType>().getShape();
  for (size_t i = 0; i < TShape.size() - 1; i++)
    result += std::to_string(TShape[i]) + "x";
  result += std::to_string(TShape[TShape.size() - 1]);
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

// FIXME: take the type from the original memref.
mlir::MemRefType getTransposedMemref(mlir::MemRefType source,
                                     llvm::ArrayRef<int64_t> permutation,
                                     mlir::Type t) {
  auto sourceMemRefShape = source.getShape();
  auto res = mlir::MemRefType::get(
      applyPermutation(sourceMemRefShape, permutation), t, {}, 0);
  return res;
}

// check that the elements in the vector are consecutive
// integer.
// {1, 2} -> ok
// {1, 1} not ok. (we cannot use std::is_sorted)
int64_t areConsecutive(llvm::ArrayRef<int64_t> indexMap) {
  bool isTrue = true;
  for (size_t i = 1; i < indexMap.size(); i++) {
    if (indexMap[i] != indexMap[i - 1] + 1) {
      isTrue = false;
      break;
    }
  }
  return isTrue;
}

llvm::SmallVector<int64_t, 8> applyIndexMap(llvm::ArrayRef<int64_t> shape,
                                            llvm::ArrayRef<int64_t> indexMap) {
  assert((shape.size() > indexMap.size()) && "shape must be > than indexMap");
  assert((indexMap[0] > 0) && "cannot applyMap to outermost dimension");

  llvm::SmallVector<int64_t, 8> result{};
  if (!areConsecutive(indexMap))
    assert(0 && "expect consecutive elements");
  int64_t newDim = 1;
  for (size_t i = 0; i < indexMap.size(); i++) {
    newDim *= shape[indexMap[i]];
  }
  for (int64_t i = 0; i < indexMap[0]; i++) {
    result.push_back(shape[i]);
  }
  result.push_back(newDim);
  return result;
}

// FIXME: take the type from the original memref.
mlir::MemRefType getReshapedMemRef(mlir::MemRefType source,
                                   llvm::ArrayRef<int64_t> indexMap,
                                   mlir::Type t) {
  auto sourceMemRefShape = source.getShape();
  auto res = mlir::MemRefType::get(applyIndexMap(sourceMemRefShape, indexMap),
                                   t, {}, 0);
  return res;
}

std::string
composeFunctionNameForMatmul(const llvm::ArrayRef<mlir::Type> &types) {
  if (types.size() != 3)
    llvm_unreachable("expect 3 memref");
  auto AShape = types[2].dyn_cast<mlir::MemRefType>().getShape();
  auto CShape = types[0].dyn_cast<mlir::MemRefType>().getShape();
  std::string result = "Matmul_";
  result += std::to_string(CShape[0]) + "x" + std::to_string(CShape[1]) + "x" +
            std::to_string(AShape[1]);
  return result;
}

std::string
composeFunctionNameForReshape(const llvm::ArrayRef<mlir::Type> &types) {
  if (types.size() != 2)
    llvm_unreachable("expect single memref");
  std::string result = "Reshape_";
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
  }
  assert(0 && "case not convered");
  return "nullptr";
}

// insert a symbol reference to "fName", inserting it into the module
// if necessary.
mlir::FlatSymbolRefAttr
getOrInsertFunction(mlir::PatternRewriter &rewriter, mlir::ModuleOp module,
                    std::string fName,
                    const llvm::ArrayRef<mlir::Type> &types) {
  auto *context = module.getContext();
  if (module.lookupSymbol(fName))
    return mlir::SymbolRefAttr::get(fName, context);
  auto libFnInfoType =
      mlir::FunctionType::get(types, {}, rewriter.getContext());
  mlir::OpBuilder::InsertionGuard guard(rewriter);
  rewriter.setInsertionPoint(module.getBody(),
                             std::prev(module.getBody()->end()));
  rewriter.create<mlir::FuncOp>(module.getLoc(), fName, libFnInfoType,
                                llvm::ArrayRef<mlir::NamedAttribute>{});
  return mlir::SymbolRefAttr::get(fName, context);
}

} // end namespace

#endif
