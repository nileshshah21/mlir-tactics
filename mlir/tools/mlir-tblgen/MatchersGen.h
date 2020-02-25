#ifndef MLIR_TACTICS_EMITTER_H
#define MLIR_TACTICS_EMITTER_H

#include "lang/Lexer.h"
#include "lang/Parser.h"
#include "mlir/TableGen/GenInfo.h"
#include "mlir/TableGen/Operator.h"

#include "mlir/TableGen/Format.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/FormatAdapters.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/TableGen/Error.h"
#include "llvm/TableGen/Record.h"
#include "llvm/TableGen/TableGenBackend.h"

namespace mlir {

namespace {
class TacticsEmitter {
public:
  TacticsEmitter(llvm::Record *tactic, raw_ostream &os);
  void emit(StringRef tacticName);

private:
  // pointer to current record.
  llvm::Record *record_;

  // pattern instantiation location
  llvm::ArrayRef<llvm::SMLoc> loc_;

  // tc parser.
  lang::Parser parser_;

  llvm::raw_ostream &os;

  // bind tensor name with emitted load matcher.
  llvm::DenseMap<StringRef, int64_t> symbolTable_;

  // counter to create unique names.
  size_t counter = 0;

  // get Location
  using identifierLine = std::pair<llvm::StringRef, unsigned>;
  std::vector<identifierLine> getLocation() const;

  // emit matching logic.
  void emitMatchLogic();

  // emit rewriting logic.
  void emitRewriteLogic();

  // emit structural logic.
  void emitStructuralMatchLogic(size_t nestedLoops);

  // emit access logic.
  using identifier = llvm::SmallSet<StringRef, 8>;
  void emitAccessMatchLogic(const lang::Comprehension &comprehension,
                            const std::pair<identifier, identifier> &ids);

  // emit operation logic.
  void emitOperationMatchLogic(const lang::Comprehension &comprehension);

  // emit mathcher for store operation.
  void emitStoreMatcherOp(const lang::Ident &ident,
                          const lang::ListView<lang::Ident> &indices);

  // emit matcher for load operation.
  void emitLoadMatcherOp(const lang::Ident &ident,
                         const lang::ListView<lang::TreeRef> &indices);

  // emit an arithmetic operation. IsRhs is assert if we are dealing with
  // the rhs operand.
  void emitArithOperationMatcher(const lang::TreeRef &t, bool isRhs = false);

  // emit a binary operation.
  void emitBinaryOperationMatcher(const lang::TreeRef &t, llvm::StringRef op);

  // emit a constant operation.
  void emitConstantOperationMatcher(const lang::Const &cst);
};
} // end namespace
} // namespace mlir
#endif
