#ifndef HELPERS_GENERATION_TACTICS_COMMON
#define HELPERS_GENERATION_TACTICS_COMMON

#include "mlir/Analysis/NestedMatcher.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Linalg/EDSC/FoldedIntrinsics.h"
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

mlir::Value createConstantFloatOp(int constant, mlir::Type t,
                                  mlir::OpBuilder &rewriter,
                                  mlir::Location &loc) {
  return rewriter.create<mlir::ConstantOp>(loc, t,
                                           rewriter.getFloatAttr(t, constant));
}

// TODO: check how we can remove this function.
mlir::Value createConstantFloatOp(mlir::Value constant, mlir::Type t,
                                  mlir::OpBuilder &rewriter,
                                  mlir::Location &loc) {
  return constant;
}

// get the LLVM dialect.
mlir::LLVM::LLVMDialect *getLLVMDialect(mlir::ModuleOp module) {
  auto *context = module.getContext();
  auto *llvmDialect = context->getRegisteredDialect<mlir::LLVM::LLVMDialect>();
  assert(llvmDialect && "expected llvm dialect to be registered");
  return llvmDialect;
}

// insert a symbol reference to "fName", inserting it into the module
// if necessary.
mlir::FlatSymbolRefAttr
getOrInsertFunction(mlir::PatternRewriter &rewriter, mlir::ModuleOp module,
                    std::string fName,
                    const llvm::ArrayRef<mlir::Type> typeOperands,
                    const llvm::ArrayRef<mlir::Type> typeResults = {}) {
  auto *context = module.getContext();
  if (module.lookupSymbol(fName))
    return mlir::SymbolRefAttr::get(fName, context);
  auto libFnInfoType =
      mlir::FunctionType::get(typeOperands, typeResults, rewriter.getContext());
  mlir::OpBuilder::InsertionGuard guard(rewriter);
  rewriter.setInsertionPoint(module.getBody(),
                             std::prev(module.getBody()->end()));
  mlir::FuncOp funcOp =
      rewriter.create<mlir::FuncOp>(module.getLoc(), fName, libFnInfoType,
                                    llvm::ArrayRef<mlir::NamedAttribute>{});
  // Insert a function attribute that it will trigger the emission of
  // _mlir_ciface_XXX.
  funcOp.setAttr("llvm.emit_c_interface",
                 mlir::UnitAttr::get(module.getContext()));
  return mlir::SymbolRefAttr::get(fName, context);
}

// return a value representing the access into a global array with
// name "name", create the array if necessary.
mlir::Value getOrCreateGlobalArray(mlir::Location loc, mlir::OpBuilder &builder,
                                   llvm::StringRef name,
                                   llvm::ArrayRef<int> values,
                                   mlir::ModuleOp module,
                                   mlir::LLVM::LLVMDialect *llvmDialect) {
  mlir::LLVM::GlobalOp global;
  if (!(global = module.lookupSymbol<mlir::LLVM::GlobalOp>(name))) {
    mlir::OpBuilder::InsertionGuard insertGuard(builder);
    builder.setInsertionPointToStart(module.getBody());
    auto type = mlir::LLVM::LLVMType::getArrayTy(
        mlir::LLVM::LLVMType::getInt32Ty(llvmDialect), values.size());
    global = builder.create<mlir::LLVM::GlobalOp>(
        loc, type, true, mlir::LLVM::Linkage::Internal, name,
        builder.getI32VectorAttr(values));
  }
  mlir::Value globalPtr = builder.create<mlir::LLVM::AddressOfOp>(loc, global);
  return globalPtr;
}

} // end namespace

#endif
