#ifndef MLIR_MATCHERS_BINARY_OP_H
#define MLIR_MATCHERS_BINARY_OP_H

#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/StandardTypes.h"

namespace mlir {

namespace detail {

/// Matcher for specific binary operation.
template <typename OpType, typename LHS_t, typename RHS_t>
struct BinaryOpMatcher {
  LHS_t lhs;
  RHS_t rhs;

  BinaryOpMatcher(LHS_t lhs, RHS_t rhs) : lhs(lhs), rhs(rhs) {}

  bool match(Operation *op) {
    if ((!isa<OpType>(op)) || (op->getNumOperands() != 2))
      return false;
    return ((lhs.match(op->getOperand(0)) && rhs.match(op->getOperand(1))) ||
            (op->isCommutative() && (lhs.match(op->getOperand(1))) &&
             (rhs.match(op->getOperand(0)))));
  }
  bool match(Value v) {
    if (auto defOp = v.getDefiningOp())
      return match(defOp);
    return false;
  }
};
} // end namespace detail

namespace matchers {
template <typename LHS_t, typename RHS_t> auto m_AddF(LHS_t lhs, RHS_t rhs) {
  return detail::BinaryOpMatcher<mlir::AddFOp, LHS_t, RHS_t>(lhs, rhs);
}
template <typename LHS_t, typename RHS_t> auto m_AddI(LHS_t lhs, RHS_t rhs) {
  return detail::BinaryOpMatcher<mlir::AddIOp, LHS_t, RHS_t>(lhs, rhs);
}
template <typename LHS_t, typename RHS_t> auto m_MulF(LHS_t lhs, RHS_t rhs) {
  return detail::BinaryOpMatcher<mlir::MulFOp, LHS_t, RHS_t>(lhs, rhs);
}
template <typename LHS_t, typename RHS_t> auto m_MulI(LHS_t lhs, RHS_t rhs) {
  return detail::BinaryOpMatcher<mlir::MulIOp, LHS_t, RHS_t>(lhs, rhs);
}
} // end namespace matchers

} // end namespace mlir

#endif
