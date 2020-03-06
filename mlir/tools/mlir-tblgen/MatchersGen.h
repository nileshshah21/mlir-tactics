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
  bool lookup(std::string key, std::string &value) const;
  bool lookup(std::string key) const;
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

  // get field with id "id". In the long run
  // if the classes in tablegen get more complicated
  // we can have simple wrapper around them.
  std::vector<StringRef> getField(StringRef id);

  void emitPreamble();
  void emitPostamble();

  // matmul builders/helpers.
  void emitMatmul();
  void emitMatmulHelpers();
  SmallVector<std::string, 3> getMatmulOperand();
  void emitMatmulBlas();

  // transpose builders/helpers.
  void emitTranspose(std::string emittedVar);
  void emitTransposeHelpers();
  std::string getTransposeOperand();
  void emitTransposeBlas(std::string emittedVar);

  // reshape builders/helpers.
  void emitReshape(std::string emittedVar);
  void emitReshapeBlas(std::string emittedVar);
  std::string getReshapeOperand();

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

  // emit matcher for load operation.
  template <typename T>
  void emitLoadOrStoreMatcherOp(const lang::Ident &ident,
                                const lang::ListView<T> &indices,
                                StringRef operation);

  // emit an arithmetic operation. IsRhs is assert if we are dealing with
  // the rhs operand
  void emitArithOperationMatcher(const lang::Comprehension &comprehension);
  void emitArithOperationMatcher(const lang::TreeRef &t);

  // emit a binary operation.
  void emitBinaryOperationMatcher(const lang::TreeRef &t, llvm::StringRef op);

  // emit a constant operation.
  void emitConstantOperationMatcher(const lang::Const &cst);
};

// TODO: handle expressions (i.e., 2*i + 1.)
template <typename T>
void TacticsEmitter::emitLoadOrStoreMatcherOp(const lang::Ident &ident,
                                              const lang::ListView<T> &indices,
                                              StringRef operation) {
  auto var = symbolTable_.getNextVariable();
  if (operation.equals("mlir::AffineLoadOp"))
    symbolTable_.insert(ident.name(), var); // A -> var0
  os.indent(8) << "auto ";
  os << var << " = m_Op<" << operation << ">(";
  os << "_" << ident.name() << "({";
  for (size_t i = 0; i < indices.size(); i++) {
    if (i == indices.size() - 1) {
      os << "_";
      os << lang::Ident(indices[indices.size() - 1]).name();
      os << "}));";
    } else
      os << "_" << lang::Ident(indices[i]).name() << ", ";
  }
}
} // end namespace
} // namespace mlir
#endif
