#ifndef MLIR_IR_ACCESS_MATCHER_H_
#define MLIR_IR_ACCESS_MATCHER_H_

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
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
      : ctx_(ctx), expr_(AffineExpr()), constant_(0), coefficient_(1) {
    bindDims(ctx, expr_, 0);
  };
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
  // additional ids in case of operators overloading
  // within placeholders. A better abstraction is
  // needed i.e., `placeholderGroup`.
  SmallVector<size_t, 4> additionalIds_;
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

inline m_Placeholder operator+(m_Placeholder p, int64_t i) {
  p.pattern_.expr_ = p.pattern_.expr_ + i;
  return p;
}

inline m_Placeholder operator-(m_Placeholder p, int64_t i) {
  p.pattern_.expr_ = p.pattern_.expr_ - i;
  return p;
}

inline m_Placeholder operator*(int64_t i, m_Placeholder p) {
  if (i <= 0)
    llvm_unreachable("Invalid coefficient for Placeholder");
  p.pattern_.expr_ = p.pattern_.expr_ * i;
  return p;
}

inline m_Placeholder operator*(m_Placeholder p, int64_t i) {
  if (i <= 0)
    llvm_unreachable("Invalid coefficient for Placeholder");
  p.pattern_.expr_ = p.pattern_.expr_ * i;
  return p;
}

inline m_Placeholder operator+(m_Placeholder p, m_Placeholder p1) {
  if (p1.id_ == p.id_) {
    p.pattern_.expr_ = p.pattern_.expr_ + p1.pattern_.expr_;
    return p;
  }
  // add dimension.
  auto ctx = p.pattern_.expr_.getContext();
  auto pos = p.pattern_.expr_.cast<AffineDimExpr>().getPosition();
  AffineExpr remapped;
  bindDims(ctx, remapped, static_cast<int>(++pos));
  remapped = p1.pattern_.expr_.replaceDimsAndSymbols({remapped}, {});
  p.pattern_.expr_ = p.pattern_.expr_ + remapped;
  p.additionalIds_.push_back(p1.id_);
  return p;
}

/// An arrayPlaceholder.
class StructuredArrayPlaceholder : public m_Placeholder {
public:
  StructuredArrayPlaceholder() : placeholders_({}){};
  StructuredArrayPlaceholder
  operator()(SmallVector<m_Placeholder, 4> indexings) {
    this->placeholders_.clear();
    this->placeholders_ = indexings;
    return *this;
  }

public:
  SmallVector<m_Placeholder, 4> placeholders_;
};
using m_ArrayPlaceholder = StructuredArrayPlaceholder;

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

// Once we know the placeholder position we re-map
// the map dimension, starting from pos `pos`. Before
// the remapping the map dimension do not have any meaning.
inline AffineExpr makeAffineExpr(MLIRContext *ctx, AffineExpr &expr, int &pos) {
  int numberOfInductions = 0;
  expr.walk([&numberOfInductions](AffineExpr e) {
    if (e.dyn_cast<AffineDimExpr>())
      numberOfInductions++;
  });
  SmallVector<AffineExpr, 4> inductions;
  for (int i = 0; i < numberOfInductions; i++) {
    inductions.push_back(AffineExpr());
    bindDims(ctx, inductions[i], pos);
    pos++;
  }
  return expr.replaceDimsAndSymbols(inductions, {});
}

template <typename OpClass> class op_load_store_matcher {
public:
  SmallVector<m_Placeholder, 4> placeholders_;
  op_load_store_matcher(SmallVector<m_Placeholder, 4> ps) : placeholders_(ps) {
    int pos = 0;
    for (auto &placeholder : placeholders_) {
      // At this point we know the placeholder
      // position in the access. We re-map the
      // affine expression starting at position `pos`.
      auto ctx = placeholder.context()->getContext();
      auto remapped = makeAffineExpr(ctx, placeholder.pattern_.expr_, pos);
      placeholder.pattern_.expr_ = remapped;
    }
  };
  op_load_store_matcher() = delete;
  bool match(Operation *op);

private:
  bool matchLoadOrStoreOp(OpClass op);

  bool matchStoreOpInAffine(AffineStoreOp op);
  bool matchLoadOpInAffine(AffineLoadOp op);
  template <typename A> bool matchLoadOrStoreOpInAffine(A op);

  bool matchStoreOpInLoop(StoreOp op);
  bool matchLoadOpInLoop(LoadOp op);
  template <typename L> bool matchLoadOrStoreOpInLoop(L op);
};

template <typename OpClass> class op_load_store_array_matcher {
public:
  StructuredArrayPlaceholder arrayPlaceholder_;
  op_load_store_array_matcher() = delete;
  op_load_store_array_matcher(StructuredArrayPlaceholder array)
      : arrayPlaceholder_(array){};
  bool match(Operation *op);

private:
  template <typename T>
  bool assingToArrayPlaceholder(T op, details::MatchingContext *ctx);
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

template <typename OpClass>
inline op_load_store_array_matcher<OpClass>
m_Op(StructuredArrayPlaceholder arg) {
  return op_load_store_array_matcher<OpClass>(arg);
}

} // end namespace matchers

} // end namespace mlir

#include "mlir/IR/Access-inl.h"

#endif // MLIR_IR_ACCESS_MATCHER_H_
