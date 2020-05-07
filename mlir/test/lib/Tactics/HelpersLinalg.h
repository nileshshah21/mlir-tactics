#ifndef HELPERS_GENERATION_TACTICS_LINALG
#define HELPERS_GENERATION_TACTICS_LINALG

#include "HelpersCommon.h"

/*
  Helpers for tablegen code generation - linalg path
*/

namespace {

mlir::Value
createLinalgReshapeOp(mlir::OpBuilder &builder, mlir::Location loc,
                      mlir::Value input,
                      llvm::ArrayRef<llvm::ArrayRef<int64_t>> reshapeMap,
                      mlir::Value destination) {
  assert(reshapeMap.size() == 2 && "expect two vectors");
  auto indexToRemap = reshapeMap[0];
  auto indexFree = reshapeMap[1];
  assert(indexToRemap.size() && "must be non empty");
  assert(indexFree.size() && "must be non empty");

  mlir::edsc::ScopedContext scope(builder, loc);
  auto ctx = input.getContext();
  llvm::SmallVector<mlir::AffineExpr, 4> dimToRemap;
  llvm::SmallVector<mlir::AffineExpr, 4> dimFree;

  // create affine exprs using the position
  // specified in the 'indexToRemap' and 'indexFree'
  // arrays.
  size_t size = indexToRemap.size();
  for (size_t i = 0; i < size; i++) {
    mlir::AffineExpr expr;
    bindDims(ctx, expr, static_cast<int>(indexToRemap[i]));
    dimToRemap.push_back(expr);
  }
  size = indexFree.size();
  for (size_t i = 0; i < size; i++) {
    mlir::AffineExpr expr;
    bindDims(ctx, expr, static_cast<int>(indexFree[i]));
    dimFree.push_back(expr);
  }

  // check if the destination is not null, if so
  // we add the destination type when building
  // the operation.
  mlir::Type destinationType = nullptr;
  if (destination)
    destinationType = destination.getType();

  // the dimension are expected to be in ascending order.
  // Thus if the reshapeMap contains the '0' dimension
  // we group {dimToRemap, {dimFree}}. If the '0' dimension
  // is *not* in reshapeMap, we group {dimFree, {dimToRemap}}.
  // TODO: find better way to express these conditions.
  if (std::find(indexToRemap.begin(), indexToRemap.end(), 0) !=
      indexToRemap.end()) {
    if (destinationType)
      return mlir::edsc::intrinsics::linalg_reshape(
          destinationType, input,
          llvm::ArrayRef<llvm::ArrayRef<mlir::AffineExpr>>{dimToRemap,
                                                           {dimFree}});
    else
      return mlir::edsc::intrinsics::linalg_reshape(
          input, llvm::ArrayRef<llvm::ArrayRef<mlir::AffineExpr>>{dimToRemap,
                                                                  {dimFree}});
  }
  if (destinationType)
    return mlir::edsc::intrinsics::linalg_reshape(
        destinationType, input,
        llvm::ArrayRef<llvm::ArrayRef<mlir::AffineExpr>>{dimFree,
                                                         {dimToRemap}});
  return mlir::edsc::intrinsics::linalg_reshape(
      input,
      llvm::ArrayRef<llvm::ArrayRef<mlir::AffineExpr>>{dimToRemap, {dimFree}});
}

} // end namespace

#endif
