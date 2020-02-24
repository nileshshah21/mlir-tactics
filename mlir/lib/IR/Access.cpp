#include "mlir/IR/Access.h"

using namespace mlir;
using namespace matchers;
using namespace details;

thread_local size_t m_Placeholder::nextId_ = 0;

LogicalResult MatchingContext::getValueForId(size_t placeholderId,
                                             Value &value) const {
  auto it = placeholderMap_.find(placeholderId);
  if (it == placeholderMap_.end())
    return failure();
  value = it->second;
  return success();
}

MatchingContext *&m_Placeholder::context() {
  thread_local MatchingContext *context = nullptr;
  return context;
}

void MatchingContext::registerPlaceholder(size_t placeholderId) {
  placeholderMap_.insert({placeholderId, nullptr});
}

void MatchingContext::dump() {
  llvm::outs() << "dumping current context..\n";
  for (const auto &it : placeholderMap_) {
    llvm::outs() << "first: " << it.first << "\n";
    if (it.second) {
      llvm::outs() << "second: " << &it.second << "\n";
    } else
      llvm::outs() << "second: nullptr"
                   << "\n";
    llvm::outs() << "---\n";
  }
}

LogicalResult MatchingContext::assignToPlaceholder(Value &val,
                                                   size_t placeholderId) {
  auto it = placeholderMap_.find(placeholderId);
  assert(it != placeholderMap_.end() && "placeholder not registered");

  if (!it->second) {
    it->second = val;
    return success();
  }
  // Placeholder with id "placeholderId" assigned to a
  // given value "val". If we want to assing another value
  // to the same id "placeholderId", the match must exit false.
  if (it->second != val)
    return failure();
  return success();
}

void m_Placeholder::dump() {
  llvm::outs() << "dumping placeholder..\n";
  llvm::outs() << id_;
}

AffinePattern m_Placeholder::buildDefaultAffinePattern() {
  assert(context() != nullptr && "expect initialized context");
  auto ctx = context()->getContext();
  assert(ctx != nullptr && "expected initialized context");
  return details::AffinePattern(ctx);
}

m_Placeholder::m_Placeholder()
    : pattern_(buildDefaultAffinePattern()), id_(nextId_++) {
  assert(context() != nullptr && "expect initialized context");
  auto ctx = context();
  ctx->registerPlaceholder(id_);
}

Value AccessPatternContext::operator[](const m_Placeholder &pl) const {
  Value value = nullptr;
  auto result = matchingContext_.getValueForId(pl.id_, value);
  if (failed(result))
    assert(0 && "placeholder not found");
  return value;
}
