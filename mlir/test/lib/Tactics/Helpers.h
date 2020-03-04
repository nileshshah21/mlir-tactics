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

template <typename... Args>
std::string composeFunctionCallName(FUNCTION id, const Args... args) {
  llvm::ArrayRef<mlir::Type> types = {args...};
  switch (id) {
  case FUNCTION::MATMUL:
    return composeFunctionNameForMatmul(types);
  case FUNCTION::RESHAPE:
    return "nullptr";
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
