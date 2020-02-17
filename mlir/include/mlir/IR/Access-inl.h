
namespace mlir {

namespace matchers {

thread_local size_t m_Placeholder::nextId_ = 0;

template <typename OpClass>
bool op_load_store_matcher<OpClass>::matchLoadOp(AffineLoadOp &op) {
  size_t dims = op.getAffineMap().getNumResults();
  if (dims != placeholders_.size())
    return false;
  SmallVector<Value, 4> operands = op.getMapOperands();
  for (size_t dim = 0; dim < dims; dim++) {
    AffineExpr affine = op.getAffineMap().getResult(dim);
    // check the affine expression.
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

template <typename OpClass>
bool op_load_store_matcher<OpClass>::matchStoreOp(AffineStoreOp &op) {
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

template <typename OpClass>
bool op_load_store_matcher<OpClass>::match(Operation *op) {
  assert(placeholders_.size() != 0 && "expect non-empty placeholders");
  if ((!op) || (!isa<OpClass>(op)))
    return false;
  if (auto loadOp = dyn_cast<AffineLoadOp>(op))
    return matchLoadOp(loadOp);
  if (auto storeOp = dyn_cast<AffineStoreOp>(op))
    return matchStoreOp(storeOp);
  assert(0 && "expect AffineStoreOp or AffineLoadOp");
  return false;
}

} // namespace matchers

} // end namespace mlir
