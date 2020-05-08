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
  auto indexPartitionOne = reshapeMap[0];
  auto indexPartitionTwo = reshapeMap[1];
  assert(indexPartitionOne.size() && "must be non empty");
  assert(indexPartitionTwo.size() && "must be non empty");

  mlir::edsc::ScopedContext scope(builder, loc);
  auto ctx = input.getContext();
  llvm::SmallVector<mlir::AffineExpr, 4> dimPartitionOne;
  llvm::SmallVector<mlir::AffineExpr, 4> dimPartitionTwo;

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

  // check if the destination is not null, if so
  // we add the destination type when building
  // the operation.
  mlir::Type destinationType = nullptr;
  if (destination)
    destinationType = destination.getType();

  // the dimension are expected to be in ascending order.
  // Thus if the reshapeMap contains the '0' dimension
  // we group {dimPartitionOne, {dimPartitionTwo}}. If the '0' dimension
  // is *not* in reshapeMap, we group {dimPartitionTwo, {dimPartitionOne}}.
  // TODO: find better way to express these conditions.
  if (std::find(indexPartitionOne.begin(), indexPartitionOne.end(), 0) !=
      indexPartitionOne.end()) {
    if (destinationType)
      return mlir::edsc::intrinsics::linalg_reshape(
          destinationType, input,
          llvm::ArrayRef<llvm::ArrayRef<mlir::AffineExpr>>{dimPartitionOne,
                                                           {dimPartitionTwo}});
    else
      return mlir::edsc::intrinsics::linalg_reshape(
          input, llvm::ArrayRef<llvm::ArrayRef<mlir::AffineExpr>>{
                     dimPartitionOne, {dimPartitionTwo}});
  }
  if (destinationType)
    return mlir::edsc::intrinsics::linalg_reshape(
        destinationType, input,
        llvm::ArrayRef<llvm::ArrayRef<mlir::AffineExpr>>{dimPartitionTwo,
                                                         {dimPartitionOne}});
  return mlir::edsc::intrinsics::linalg_reshape(
      input, llvm::ArrayRef<llvm::ArrayRef<mlir::AffineExpr>>{
                 dimPartitionOne, {dimPartitionTwo}});
}

} // end namespace

#endif
