#ifndef HELPERS_GENERATION_TACTICS_COMMON
#define HELPERS_GENERATION_TACTICS_COMMON

#include "mlir/Analysis/NestedMatcher.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
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
  rewriter.create<mlir::FuncOp>(module.getLoc(), fName, libFnInfoType,
                                llvm::ArrayRef<mlir::NamedAttribute>{});
  return mlir::SymbolRefAttr::get(fName, context);
}

// return a value representing the access into a global array with
// name "name", create the array if necessary.
mlir::Value getOrCreateGlobalArray(mlir::Location loc, mlir::OpBuilder &builder,
                                   llvm::StringRef name,
                                   const llvm::ArrayRef<int> &values,
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
