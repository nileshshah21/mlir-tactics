#ifndef MLIR_TACTICS_EMITTER_H
#define MLIR_TACTICS_EMITTER_H

#include "lang/Lexer.h"
#include "lang/Parser.h"
#include "mlir/TableGen/GenInfo.h"
#include "mlir/TableGen/Operator.h"

#include "mlir/TableGen/Format.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/Support/FormatAdapters.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/TableGen/Error.h"
#include "llvm/TableGen/Record.h"
#include "llvm/TableGen/TableGenBackend.h"
#include <string>

namespace mlir {

namespace {
class SymbolTableMap {
public:
  SymbolTableMap() : nextId_(0){};
  std::string getNextVariable();
  void insert(std::string key, std::string value);
  void updateOrInsert(std::string key, std::string value);
  std::string lookup(std::string key) const;
  void clear();
  void dump() const;

private:
  size_t nextId_;
  std::map<std::string, std::string> symbolTable_;
};

class BuilderEmitter {
public:
  BuilderEmitter(llvm::Record *builder, raw_ostream &os);
  void emit();

private:
  llvm::Record *record_;
  llvm::raw_ostream &os;

  void emitMatmul();
  void emitMatmulHelpers();

  std::string emitTranspose();
  std::string emitTransposeHelpers();

  void emitErase();

public:
  // FIXME make friend with TacticsEmitter instead of public.
  static thread_local SymbolTableMap symbolTable_;
};

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

  // symbol table map.
  SymbolTableMap symbolTable_;

  // get Location
  using identifierLine = std::pair<llvm::StringRef, unsigned>;
  std::vector<identifierLine> getLocation() const;

  // emit matching logic.
  using identifier = llvm::SmallSet<std::string, 8>;
  void emitMatchLogic(const lang::Comprehension &comprehension,
                      const std::pair<identifier, identifier> &ids);

  // emit rewriting logic.
  void emitRewriteLogic();

  // emit structural logic.
  void emitStructuralMatchLogic(size_t nestedLoops);

  // emit access logic.
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
