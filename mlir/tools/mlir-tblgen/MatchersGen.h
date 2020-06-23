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

// Walk tree (see teckly).
void walkTree(const lang::TreeRef &tree,
              std::function<void(const lang::TreeRef &)> fn) {
  fn(tree);
  for (auto e : tree->trees())
    walkTree(e, fn);
}

// TODO: make matmulEntryBlas and MatVecEntryBlas
// derivate classes of a common base one.
// handy function to manipulate
// tablegen entries for the matmul builder.
class MatmulBlasEntry {
public:
  explicit MatmulBlasEntry(llvm::Record *record) : record_(record) {}
  llvm::StringRef alpha() const;
  llvm::StringRef beta() const;
  int64_t dimensionForM() const;
  int64_t dimensionForN() const;
  int64_t dimensionForK() const;
  llvm::StringRef transA() const;
  llvm::StringRef transB() const;
  std::vector<llvm::StringRef> inputs() const;
  llvm::StringRef outputs() const;

private:
  llvm::Record *record_;
};

class MatvecBlasEntry {
public:
  explicit MatvecBlasEntry(llvm::Record *record) : record_(record) {}
  llvm::StringRef transA() const;
  llvm::StringRef alpha() const;
  llvm::StringRef beta() const;
  std::vector<llvm::StringRef> inputs() const;
  llvm::StringRef outputs() const;

private:
  llvm::Record *record_;
};

class ReshapeBlasEntry {
public:
  explicit ReshapeBlasEntry(llvm::Record *record) : record_(record) {}
  llvm::StringRef map() const;
  llvm::StringRef inputs() const;
  llvm::StringRef outputs() const;

private:
  llvm::Record *record_;
};

class TransposeBlasEntry {
public:
  explicit TransposeBlasEntry(llvm::Record *record) : record_(record) {}
  llvm::StringRef permutation() const;
  llvm::StringRef inputs() const;
  llvm::StringRef outputs() const;

private:
  llvm::Record *record_;
};

enum class Target { CPU, GPU };

// TODO: Handle coeff. and increment.
class Index {
public:
  Index() = delete;
  Index(std::string id) : id_(id){};
  std::string getId() const { return id_; };

private:
  std::string id_;
};

class Tensor {
public:
  Tensor() = delete;
  Tensor(std::string id, std::vector<Index> indexes)
      : id_(id), indexes_(indexes){};
  std::string getId() const { return id_; };
  std::vector<Index> getIndexes() const { return indexes_; };
  friend bool operator<(const Tensor &t1, const Tensor &t2);
  template <typename T>
  static Tensor buildTensor(const lang::Ident &ident,
                            const lang::ListView<T> &indices);

private:
  std::string id_;
  std::vector<Index> indexes_;
};

// sorting operator for Tensor class used by std::map.
bool operator<(const Tensor &t1, const Tensor &t2) {
  auto t1S = t1.getId();
  auto t2S = t2.getId();
  auto t1Indexes = t1.getIndexes();
  auto t2Indexes = t2.getIndexes();
  for (const auto &index : t1Indexes)
    t1S += index.getId();
  for (const auto &index : t2Indexes)
    t2S += index.getId();
  return t1S < t2S;
}

template <typename T> std::string getInductionVariable(T &tree) {
  if (std::is_same<lang::Ident, T>::value)
    return lang::Ident(tree).name();
  // if we have multiple induction variables
  // for an access (i.e., A(i + j)) we chain
  // i and j together -> ij
  std::string indVars = "";
  walkTree(tree, [&](const lang::TreeRef t) {
    if (t->kind() == lang::TK_IDENT)
      indVars += lang::Ident(t).name();
  });
  return indVars;
}

template <typename T>
Tensor Tensor::buildTensor(const lang::Ident &ident,
                           const lang::ListView<T> &indices) {
  auto tensorId = ident.name();
  std::vector<Index> tensorIndexes{};
  for (const auto &index : indices)
    tensorIndexes.push_back(Index(getInductionVariable(index)));
  return Tensor(tensorId, tensorIndexes);
}

template <typename T> class SymbolTableMap {
public:
  SymbolTableMap() : nextId_(0){};
  std::string getNextVariable();
  void insert(T key, std::string value);
  void updateOrInsert(T key, std::string value);
  bool lookup(T, std::string &value) const;
  bool lookup(T key) const;
  void clear();
  void dump() const;

private:
  size_t nextId_;
  std::map<T, std::string> symbolTable_;
};

template <typename T> void SymbolTableMap<T>::insert(T key, std::string value) {
  symbolTable_.insert(std::make_pair(key, value));
}

template <typename T>
bool SymbolTableMap<T>::lookup(T key, std::string &value) const {
  auto it = symbolTable_.find(key);
  if (it != symbolTable_.end()) {
    value = it->second;
    return true;
  }
  return false;
}

template <typename T> bool SymbolTableMap<T>::lookup(T key) const {
  std::string dummyCapture;
  return lookup(key, dummyCapture);
}

template <typename T>
void SymbolTableMap<T>::updateOrInsert(T key, std::string value) {
  auto it = symbolTable_.find(key);
  if (it == symbolTable_.end())
    insert(key, value);
  else
    it->second = value;
}

template <typename T> void SymbolTableMap<T>::clear() { symbolTable_.clear(); }

template <typename T> std::string SymbolTableMap<T>::getNextVariable() {
  std::string var = "var" + std::to_string(nextId_++);
  return var;
}

struct MatvecTy {
  std::string output;
  std::vector<std::string> inputs;
  std::string alpha;
  std::string beta;
  bool transA;
};

struct MatmulTy {
  std::string output;
  std::vector<std::string> inputs;
  std::string alpha;
  std::string beta;

  bool transA;
  bool transB;

  int dimForM;
  int dimForN;
  int dimForK;
};

class BuilderEmitter {
public:
  BuilderEmitter(llvm::Record *builder, bool lastBeforeEraseOp,
                 raw_ostream &os);
  void emit();

private:
  llvm::Record *record_;
  // this flag tells the builder
  // if he is the last one before
  // eraseOp. It is necessary in the linalg
  // path to avoid DCE by emitting a copyOp.
  bool lastBeforeEraseOp_;
  llvm::raw_ostream &os;

  // lookup operands in symbol table.
  std::string lookUpOperand(llvm::StringRef operand) const;
  std::vector<std::string>
  lookUpOperands(std::vector<llvm::StringRef> operands) const;

  void emitPreamble(bool &isEmitted, std::string &dest, llvm::StringRef output);
  void emitPostamble();

  // matmul builders/helpers.
  void emitMatmul(bool isEmitted, std::string destBuff);
  void emitMatmulLinalg(MatmulTy &mmi);
  void emitMatmulBlas(MatmulTy &mmi, Target t);

  // transpose builders/helpers.
  void emitTranspose(bool isEmitted, std::string destBuff);
  void emitTransposeBlas(bool isEmitted, std::string destBuff,
                         std::string input, std::string permutation);
  void emitTransposeLinalg(std::string destBuff, std::string input,
                           std::string permuation);

  // reshape builders/helpers.
  void emitReshape(bool isEmitted, std::string destBuff);
  void emitReshapeBlas(bool isEmitted, std::string destBuff, std::string input,
                       std::string permutation);
  void emitReshapeLinalg(std::string destBuff, std::string input,
                         std::string permutation);

  // matvec builders/helpers.
  void emitMatvec(bool isEmitted, std::string destBuff);
  void emitMatvecBlas(MatvecTy &matvecInfo);
  void emitMatvecLinalg(MatvecTy &matvecInfo);

  void emitErase();

public:
  // FIXME make friend with TacticsEmitter instead of public.
  static thread_local SymbolTableMap<std::string> symbolTable_;
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
  SymbolTableMap<Tensor> symbolTable_;

  // get Location
  using identifierLine = std::pair<llvm::StringRef, unsigned>;
  std::vector<identifierLine> getLocation() const;

  // emit matching logic.
  using identifierIterators = llvm::SmallSet<std::string, 8>;
  using identifierTensors = llvm::SmallSet<std::pair<bool, std::string>, 8>;
  void
  emitMatchLogic(const lang::Comprehension &comprehension,
                 const std::pair<identifierIterators, identifierTensors> &ids);

  // emit rewriting logic.
  void emitRewriteLogic();

  // emit structural logic.
  void emitStructuralMatchLogic(size_t nestedLoops);

  // emit access logic.
  void emitAccessMatchLogic(
      const lang::Comprehension &comprehension,
      const std::pair<identifierIterators, identifierTensors> &ids);

  // emit operation logic.
  void emitOperationMatchLogic(const lang::Comprehension &comprehension);

  // emit matcher for load operation.
  template <typename T>
  void emitLoadOrStoreMatcherOp(const lang::Ident &ident,
                                const lang::ListView<T> &indices,
                                StringRef operation);
  template <typename T> void emitTree(const T &tree);

  // emit an arithmetic operation. IsRhs is assert if we are dealing with
  // the rhs operand
  void emitArithOperationMatcher(const lang::Comprehension &comprehension);
  void emitArithOperationMatcher(const lang::TreeRef &t);

  // emit a binary operation.
  void emitBinaryOperationMatcher(const lang::TreeRef &t, llvm::StringRef op);

  // emit a constant operation.
  void emitConstantOperationMatcher(const lang::Const &cst);
};

template <typename T> void TacticsEmitter::emitTree(const T &tree) {
  if (std::is_same<lang::Ident, T>::value) {
    os << "_" << lang::Ident(tree).name();
    return;
  }
  // TODO make me generic.
  bool printPlusAfterIdent = false;
  walkTree(tree, [&](const lang::TreeRef t) {
    if (t->kind() == '+') {
      printPlusAfterIdent = true;
    }
    if (t->kind() == lang::TK_IDENT) {
      os << "_" << lang::Ident(t).name();
      if (printPlusAfterIdent) {
        os << " + ";
        printPlusAfterIdent = false;
      }
    }
  });
}

// TODO: handle expressions (i.e., 2*i + 1.)
template <typename T>
void TacticsEmitter::emitLoadOrStoreMatcherOp(const lang::Ident &ident,
                                              const lang::ListView<T> &indices,
                                              StringRef operation) {
  auto var = symbolTable_.getNextVariable();
  if (operation.equals("mlir::AffineLoadOp"))
    symbolTable_.insert(Tensor::buildTensor(ident, indices), var); // A -> var0
  os.indent(8) << "auto ";
  os << var << " = m_Op<" << operation << ">(";
  os << "_" << ident.name() << "({";
  for (size_t i = 0; i < indices.size(); i++) {
    emitTree(indices[i]);
    if (i == indices.size() - 1)
      os << "}));";
    else
      os << ", ";
  }
}
} // end namespace
} // namespace mlir
#endif
