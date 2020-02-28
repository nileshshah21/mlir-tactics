#include <iostream>
namespace mlir {

namespace matchers {

template <typename OpClass>
template <typename A>
bool op_load_store_matcher<OpClass>::matchLoadOrStoreOpInAffine(A op) {
  size_t dims = op.getAffineMap().getNumResults();
  if (dims != placeholders_.size())
    return false;
  SmallVector<Value, 4> operands = op.getMapOperands();
  for (size_t dim = 0; dim < dims; dim++) {
    AffineExpr affine = op.getAffineMap().getResult(dim);
    // check affine expression.
    if (placeholders_[dim].pattern_.expr_ != affine)
      return false;
    auto matchingContext = placeholders_[dim].context();
    auto isValid = matchingContext->assignToPlaceholder(operands[dim],
                                                        placeholders_[dim].id_);
    if (failed(isValid))
      return false;
  }
  return true;
}

// TODO: check how to get coeff. and inc. information for the index.
template <typename OpClass>
template <typename L>
bool op_load_store_matcher<OpClass>::matchLoadOrStoreOpInLoop(L op) {
  auto operands = op.getIndices();
  auto dims = operands.size();
  if (dims != placeholders_.size())
    return false;
  auto matchingContext = placeholders_[0].context();
  for (size_t dim = 0; dim < dims; dim++) {
    Value operandAtPos = operands[dim];
    auto isValid = matchingContext->assignToPlaceholder(operandAtPos,
                                                        placeholders_[dim].id_);
    if (failed(isValid))
      return false;
  }
  return true;
}

template <typename OpClass>
bool op_load_store_matcher<OpClass>::matchStoreOpInAffine(AffineStoreOp op) {
  return matchLoadOrStoreOpInAffine<AffineStoreOp>(op);
}

template <typename OpClass>
bool op_load_store_matcher<OpClass>::matchLoadOpInAffine(AffineLoadOp op) {
  return matchLoadOrStoreOpInAffine<AffineLoadOp>(op);
}

template <typename OpClass>
bool op_load_store_matcher<OpClass>::matchLoadOpInLoop(LoadOp op) {
  return matchLoadOrStoreOpInLoop<LoadOp>(op);
}

template <typename OpClass>
bool op_load_store_matcher<OpClass>::matchStoreOpInLoop(StoreOp op) {
  return matchLoadOrStoreOpInLoop<StoreOp>(op);
}

template <typename OpClass>
bool op_load_store_matcher<OpClass>::matchLoadOrStoreOp(OpClass op) {
  if (isa<AffineStoreOp>(*op))
    return matchStoreOpInAffine(static_cast<AffineStoreOp>(op));
  if (isa<AffineLoadOp>(*op))
    return matchLoadOpInAffine(static_cast<AffineLoadOp>(op));
  if (isa<LoadOp>(*op))
    return matchLoadOpInLoop(static_cast<LoadOp>(op));
  if (isa<StoreOp>(*op))
    return matchStoreOpInLoop(static_cast<StoreOp>(op));
  llvm_unreachable("expect AffineStore/AffineLoad/LoadOp/StoreOp");
  return false;
}

template <typename OpClass>
bool op_load_store_matcher<OpClass>::match(Operation *op) {
  assert(placeholders_.size() != 0 && "expect non-empty placeholders");
  if ((!op) || (!isa<OpClass>(op)))
    return false;
  return matchLoadOrStoreOp(dyn_cast<OpClass>(op));
}

template <typename OpClass>
template <typename T>
bool op_load_store_array_matcher<OpClass>::assingToArrayPlaceholder(
    T op, details::MatchingContext *ctx) {
  auto operand = op.getMemRef();
  auto isValid = ctx->assignToPlaceholder(operand, arrayPlaceholder_.id_);
  if (failed(isValid))
    return false;
  return true;
}

template <typename OpClass>
bool op_load_store_array_matcher<OpClass>::match(Operation *op) {
  auto p = op_load_store_matcher<OpClass>(arrayPlaceholder_.placeholders_);
  if (!p.match(op))
    return false;
  auto matchingContext = arrayPlaceholder_.context();
  if (auto loadOp = dyn_cast<AffineLoadOp>(op)) {
    return assingToArrayPlaceholder<AffineLoadOp>(loadOp, matchingContext);
  }
  if (auto storeOp = dyn_cast<AffineStoreOp>(op)) {
    return assingToArrayPlaceholder<AffineStoreOp>(storeOp, matchingContext);
  }
  if (auto loadOp = dyn_cast<LoadOp>(op)) {
    return assingToArrayPlaceholder<LoadOp>(loadOp, matchingContext);
  }
  if (auto storeOp = dyn_cast<StoreOp>(op)) {
    return assingToArrayPlaceholder<StoreOp>(storeOp, matchingContext);
  }
  llvm_unreachable("expect AffineStore/AffineLoad/LoadOp/StoreOp");
  return false;
}

} // namespace matchers

} // end namespace mlir
