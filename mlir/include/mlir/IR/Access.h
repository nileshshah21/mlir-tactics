#ifndef MLIR_IR_ACCESS_MATCHER_H_
#define MLIR_IR_ACCESS_MATCHER_H_

#include "mlir/Dialect/AffineOps/AffineOps.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/Support/raw_ostream.h" // remove me

/* Typical use:

{
  AccessPatternContext pctx(ctx);
  auto _i = m_Placeholder();
  auto _j = m_Placeholder();
  auto _k = m_Placeholder();
  auto a = m_Op<AffineLoadOp>(_i, _k);
  auto b = m_Op<AffineLoadOp>(_k, _j);
  auto c = m_Op<AffineLoadOp>(_i, _j);
  auto p1 = m_Op<AddFOp>(c, m_Op<MulFOp>(a, b));
  match(p1, ...)
  auto matchedI = pctx[_i];
}

*/

namespace mlir {

using namespace mlir;

namespace details {

/// An affine access pattern.
class AffinePattern {
public:
  MLIRContext *ctx_;
  AffineExpr expr_;
  int64_t constant_;
  int64_t coefficient_;

public:
  AffinePattern() = delete;
  AffinePattern(MLIRContext *ctx)
      : ctx_(ctx), expr_(AffineExpr()), constant_(0), coefficient_(1){};
};

/// The matching context. It keeps track of what a given placeholder has already
/// matched. It has a "global" view of what is going on. Each placeholder when
/// instantiated register itself to the context passing its unique id. During
/// the matching, a placeholder asks the context if it can match the underneath
/// Value (i.e., not already assigned to another placeholder).
class MatchingContext {
public:
  MatchingContext() = delete;
  MatchingContext(MLIRContext *ctx) : ctx_(ctx) { placeholderMap_.init(8); };

  /// dump the current context.
  void dump();

  /// Get underneath context.
  MLIRContext *getContext() { return ctx_; };

  /// Allow registering a placeholder.
  void registerPlaceholder(size_t placeholderId);

  /// Check if the matching between placeholder "placeholderId" and
  /// value "val" is ok.
  LogicalResult assignToPlaceholder(Value &val, size_t placeholderId);

  /// Get Value for id "placeholderId"
  LogicalResult getValueForId(size_t placeholderId, Value &value) const;

private:
  MLIRContext *ctx_;
  llvm::DenseMap<size_t, Value> placeholderMap_;
};

} // end namespace details

namespace matchers {

/// A placeholder.
class m_Placeholder {
public:
  details::AffinePattern pattern_;
  // non-const to allow default assignement operator.
  size_t id_;
  // FIXME: make me private.
  static details::MatchingContext *&context();

private:
  friend class AccessPatternContext;
  static thread_local size_t nextId_;

public:
  m_Placeholder();
  m_Placeholder(const m_Placeholder &) = default;
  void dump();

private:
  details::AffinePattern buildDefaultAffinePattern();
};

/// Structure to avoid passing context information
/// to all the API functions.
class AccessPatternContext {
public:
  AccessPatternContext(MLIRContext *ctx)
      : matchingContext_(details::MatchingContext(ctx)) {
    assert(m_Placeholder::context() == nullptr &&
           "Only a single matchingContext is supported");
    m_Placeholder::context() = &matchingContext_;
  }
  ~AccessPatternContext() { m_Placeholder::context() = nullptr; }
  details::MatchingContext matchingContext_;

  Value operator[](const m_Placeholder &pl) const;
};

template <typename OpClass> class op_load_store_matcher {
public:
  SmallVector<m_Placeholder, 4> placeholders_;
  op_load_store_matcher(SmallVector<m_Placeholder, 4> ps) : placeholders_(ps) {
    int pos = 0;
    for (auto &placeholder : placeholders_) {
      // At this point we know the placeholder
      // position in the access. We create an
      // affine map to match.
      auto ctx = placeholder.context()->getContext();
      detail::bindDims(ctx, placeholder.pattern_.expr_, pos++);
      placeholder.pattern_.expr_ =
          placeholder.pattern_.expr_ + placeholder.pattern_.constant_;
      placeholder.pattern_.expr_ =
          placeholder.pattern_.expr_ * placeholder.pattern_.coefficient_;
    }
  };
  op_load_store_matcher() = delete;
  bool match(Operation *op);

private:
  bool matchStoreOp(AffineStoreOp &op);
  bool matchLoadOp(AffineLoadOp &op);
};

template <class T, class...> struct are_same : std::true_type {};

template <class T, class U, class... TT>
struct are_same<T, U, TT...>
    : std::integral_constant<bool,
                             std::is_same<T, U>{} && are_same<T, TT...>{}> {};

template <typename OpClass, typename... Args>
inline op_load_store_matcher<OpClass> m_Op(m_Placeholder arg, Args... args) {
  static_assert(are_same<m_Placeholder, Args...>{},
                "all args must be Placeholder");
  return op_load_store_matcher<OpClass>({arg, args...});
}

} // end namespace matchers

} // end namespace mlir

#include "mlir/IR/Access-inl.h"

#endif // MLIR_IR_ACCESS_MATCHER_H_
