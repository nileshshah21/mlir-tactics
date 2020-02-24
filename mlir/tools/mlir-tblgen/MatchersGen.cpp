#include "mlir/Support/STLExtras.h"
#include "mlir/TableGen/GenInfo.h"
#include "mlir/TableGen/OpClass.h"
#include "mlir/TableGen/Operator.h"

#include "mlir/TableGen/Format.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/FormatAdapters.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/Signals.h"
#include "llvm/TableGen/Error.h"
#include "llvm/TableGen/Record.h"
#include "llvm/TableGen/TableGenBackend.h"

#include "lang/Lexer.h"
#include "lang/Parser.h"
#include "llvm/Support/raw_ostream.h" // remove me.

using namespace llvm;
using namespace mlir;
using namespace lang;

using mlir::tblgen::Operator;

namespace llvm {
using identifierLine = std::pair<StringRef, unsigned>;
template <> struct format_provider<identifierLine> {
  static void format(const identifierLine &v, raw_ostream &os,
                     StringRef style) {
    os << v.first << ":" << v.second;
  }
};
} // end namespace llvm

namespace {
class TacticsEmitter {
public:
  TacticsEmitter(Record *tactic, raw_ostream &os);
  void emit(StringRef tacticName);

private:
  // pointer to current record.
  Record *record_;

  // pattern instantiation location
  ArrayRef<llvm::SMLoc> loc_;

  // tc parser.
  Parser parser_;

  raw_ostream &os;

  // bind tensor name with emitted load matcher.
  llvm::DenseMap<StringRef, int64_t> symbolTable_;

  // counter to create unique names.
  size_t counter = 0;

  // get Location
  using identifierLine = std::pair<StringRef, unsigned>;
  std::vector<identifierLine> getLocation() const;

  // emit matching logic.
  void emitMatchLogic();

  // emit rewriting logic.
  void emitRewriteLogic();

  // emit structural logic.
  void emitStructuralMatchLogic(size_t nestedLoops);

  // emit access logic.
  using identifier = SmallSet<StringRef, 8>;
  void emitAccessMatchLogic(const Comprehension &comprehension,
                            const std::pair<identifier, identifier> &ids);

  // emit operation logic.
  void emitOperationMatchLogic(const Comprehension &comprehension);

  // emit mathcher for store operation.
  void emitStoreMatcherOp(const Ident &ident, const ListView<Ident> &indices);

  // emit matcher for load operation.
  void emitLoadMatcherOp(const Ident &ident, const ListView<TreeRef> &indices);

  // emit an arithmetic operation. IsRhs is assert if we are dealing with
  // the rhs operand.
  void emitArithOperationMatcher(const TreeRef &t, bool isRhs = false);

  // emit a binary operation.
  void emitBinaryOperationMatcher(const TreeRef &t, StringRef op);

  // emit a constant operation.
  void emitConstantOperationMatcher(const Const &cst);
};

} // end namespace

// Walk tree (see teckly).
void walkTree(const TreeRef &tree, std::function<void(const TreeRef &)> fn) {
  fn(tree);
  for (auto e : tree->trees())
    walkTree(e, fn);
}

// collect iterators from comprehension.
std::pair<SmallSet<StringRef, 8>, SmallSet<StringRef, 8>>
collectIteratorsAndTensorNames(const Comprehension &comprehension) {
  SmallSet<StringRef, 8> iterators;
  SmallSet<StringRef, 8> names;

  for (const auto &lhs : comprehension.indices()) {
    iterators.insert(StringRef(lhs.name()));
  }
  names.insert(StringRef(comprehension.ident().name()));

  walkTree(comprehension.rhs(), [&](const TreeRef &t) {
    if (t->kind() == TK_APPLY) {
      auto tc = Apply(t);
      names.insert(StringRef(tc.name().name()));
      auto tcIters = tc.arguments();
      for (const auto &tcIter : tcIters) {
        if (tcIter->kind() == TK_IDENT)
          iterators.insert(StringRef(Ident(tcIter).name()));
      }
    }
  });

  return std::make_pair(iterators, names);
}

void TacticsEmitter::emitStoreMatcherOp(const Ident &ident,
                                        const ListView<Ident> &indices) {
  os.indent(8) << "auto ";
  os << "var" << counter++ << " = m_Op<mlir::AffineStoreOp>(";
  os << "_" << StringRef(ident.name()) << "({";
  for (size_t i = 0; i < indices.size(); i++) {
    if (i == indices.size() - 1) {
      os << "_" << StringRef(Ident(indices[indices.size() - 1]).name());
      os << "}));";
    } else
      os << "_" << StringRef(Ident(indices[i]).name()) << ", ";
  }
}

// TODO: handle expressions (i.e., 2*i + 1.)
// TODO: merge with emitStoreMatcherOp.
void TacticsEmitter::emitLoadMatcherOp(const Ident &ident,
                                       const ListView<TreeRef> &indices) {
  symbolTable_.insert({StringRef(ident.name()), counter});
  os.indent(8) << "auto ";
  os << "var" << counter++ << " = m_Op<mlir::AffineLoadOp>(";
  os << "_" << StringRef(ident.name()) << "({";
  for (size_t i = 0; i < indices.size(); i++) {
    if (i == indices.size() - 1) {
      os << "_" << StringRef(Ident(indices[indices.size() - 1]).name());
      os << "}));";
    } else
      os << "_" << StringRef(Ident(indices[i]).name()) << ", ";
  }
}

void TacticsEmitter::emitBinaryOperationMatcher(const TreeRef &t,
                                                StringRef op) {
  os << "m_Op<" << op << ">(";
  emitArithOperationMatcher(t->trees().at(0), false);
  emitArithOperationMatcher(t->trees().at(1), true);
  os << ")";
}

void TacticsEmitter::emitConstantOperationMatcher(const Const &cst) {
  assert(0 && "not implemented");
}

void TacticsEmitter::emitArithOperationMatcher(const TreeRef &t, bool isRhs) {
  switch (t->kind()) {
  case '*':
    return emitBinaryOperationMatcher(t, "mlir::MulFOp");
  case '-':
    return emitBinaryOperationMatcher(t, "mlir::SubFOp");
  case '+':
    return emitBinaryOperationMatcher(t, "mlir::AddFOp");
  case '/':
    return emitBinaryOperationMatcher(t, "mlir::SubFOp");
  case TK_NUMBER:
  case TK_CONST:
    return emitConstantOperationMatcher(Const(t));
  case TK_APPLY: {
    auto tc = Apply(t);
    auto tcName = Ident(tc.name());
    os << "var" << symbolTable_.lookup(StringRef(tcName.name()));
    if (!isRhs)
      os << ", ";
    return;
  }
  case TK_IDENT: {
    assert(0 && "not handled");
    return;
  }
  default:
    assert(0 && "unknown tree type");
  }
}

void TacticsEmitter::emitOperationMatchLogic(
    const Comprehension &comprehension) {
  // emit the only one store operation.
  auto assignment = comprehension.assignment();
  switch (assignment->kind()) {
  case '=':
    emitStoreMatcherOp(comprehension.ident(), comprehension.indices());
    break;
  default:
    assert(0 && "other assignment operators not implemented!");
  }

  // check if the store match the last operation.
  os << R"(
        auto store = llvm::dyn_cast<mlir::AffineStoreOp>(*std::prev(loopBody.front().end(), 2));
  )";
  os << R"(
        if (!matchPattern(store, var0))
          return false;
  )";

  os << "\n";
  // emit all the load operations.
  walkTree(comprehension.rhs(), [&](const TreeRef &t) {
    if (t->kind() == TK_APPLY) {
      auto tc = Apply(t);
      emitLoadMatcherOp(tc.name(), tc.arguments());
      os << "\n";
    }
  });
  os.indent(8) << "auto bodyMatcher = ";
  // emit operation matcher.
  emitArithOperationMatcher(comprehension.rhs());
  os << ";\n";

  auto rootOperation = comprehension.rhs();
  switch (rootOperation->kind()) {
  case '*': {
    os << R"(
        auto rootOp = 
          llvm::dyn_cast_or_null<mlir::MulFOp>(store.getValueToStore().getDefiningOp());
      )";
    break;
  }
  default:
    assert(0 && "operation not supported");
  }
  os << R"(
        if ((!rootOp) || (!bodyMatcher.match(rootOp)))
          return false;
        return true;
  )";
}

void TacticsEmitter::emitAccessMatchLogic(
    const Comprehension &comprehension,
    const std::pair<SmallSet<StringRef, 8>, SmallSet<StringRef, 8>> &ids) {
  os << R"(
      {)";
  os << R"(
        AccessPatternContext pctx(loopBody.getContext());
  )";
  os << "\n";

  for (const auto &id : ids.first) {
    os.indent(8) << "auto ";
    os << "_" << id << " = m_Placeholder();\n";
  }
  for (const auto &id : ids.second) {
    os.indent(8) << "auto ";
    os << "_" << id << " = m_ArrayPlaceholder();\n";
  }
  os << "\n";
  emitOperationMatchLogic(comprehension);
  os << R"(
      })";
}

void TacticsEmitter::emitStructuralMatchLogic(size_t nestedLoops) {
  os << R"(
    {)";
  os << R"(
      mlir::NestedPatternContext raii;
      using namespace mlir::matcher;
  )";

  os.indent(4) << "auto m = ";
  for (size_t i = 0; i < nestedLoops; i++) {
    if (i == nestedLoops - 1) {
      os << "For(body)";
      for (size_t j = i; j > 0; j--) {
        os << ")";
        if (j == 1)
          os << ";";
      }
    } else
      os << "For(";
  }

  os << R"(
      llvm::SmallVector<mlir::NestedMatch, 1> matches;
      m.match(op, &matches);
      if (matches.empty())
        return matchFailure();
    })";
}

void TacticsEmitter::emitMatchLogic() {
  auto comprehension = Comprehension(parser_.parseStmt());
  auto iteratorsAndTensorNames = collectIteratorsAndTensorNames(comprehension);
  os << R"(
    auto body = [](mlir::Operation &op) -> bool {
      auto loop = llvm::cast<mlir::AffineForOp>(op);
      mlir::Region &loopBody = loop.getLoopBody();
      using namespace mlir::matchers;
  )";
  emitAccessMatchLogic(comprehension, iteratorsAndTensorNames);
  os << R"(
    }; // end callback.
  )";
  emitStructuralMatchLogic(iteratorsAndTensorNames.second.size());
}

void TacticsEmitter::emitRewriteLogic() {
  os.indent(4);
  os << "rewriter.eraseOp(op);\n";
}

TacticsEmitter::TacticsEmitter(Record *record, raw_ostream &os)
    : record_(record), loc_(record_->getLoc()),
      parser_(Parser(record_->getValueAsString("pattern").str())), os(os){};

std::vector<std::pair<StringRef, unsigned>>
TacticsEmitter::getLocation() const {
  std::vector<std::pair<StringRef, unsigned>> result;
  result.reserve(record_->getLoc().size());
  for (auto loc : record_->getLoc()) {
    unsigned buf = llvm::SrcMgr.FindBufferContainingLoc(loc);
    assert(buf && "invalid source location");
    result.emplace_back(
        llvm::SrcMgr.getBufferInfo(buf).Buffer->getBufferIdentifier(),
        llvm::SrcMgr.getLineAndColumn(loc, buf).first);
  }
  return result;
}

void TacticsEmitter::emit(StringRef tacticName) {

  // Emit Rewrite
  auto locs = getLocation();
  os << formatv("/* Generated from:\n\t{0:$[ instantiating\n\t]}\n*/\n",
                make_range(locs.rbegin(), locs.rend()));

  os << formatv(
      R"(struct {0} : public mlir::OpRewritePattern<mlir::AffineForOp> {{)",
      tacticName);

  os << R"(
  using mlir::OpRewritePattern<mlir::AffineForOp>::OpRewritePattern;)";
  os << "\n";
  // Emit matchAndRewrite() function.
  os << R"(
  mlir::PatternMatchResult matchAndRewrite(mlir::AffineForOp op,
                                           mlir::PatternRewriter &rewriter) const override {)";
  os << "\n";
  os.indent(4) << "// Match";
  emitMatchLogic();
  os << "\n";
  os.indent(4) << "// Rewrite\n";
  emitRewriteLogic();
  os.indent(4) << "return matchSuccess();\n";
  os << "  };\n";
  os << "};\n";
}

static void emitMatchersRewriters(const RecordKeeper &records,
                                  raw_ostream &os) {
  llvm::emitSourceFileHeader("Tactics", os);

  const auto &tactics = records.getAllDerivedDefinitions("Tactics");
  auto numTactics = tactics.size();

  std::vector<std::string> tacticsNames;
  tacticsNames.reserve(numTactics);

  std::string baseTacticsName = "tactic";
  int tacticsIndex = 0;

  for (auto &tactic : tactics) {
    std::string name = baseTacticsName + llvm::utostr(tacticsIndex++);
    TacticsEmitter(tactic, os).emit(name);
    tacticsNames.push_back(std::move(name));
  }

  // Emit function to add the generated matchers to the pattern list.
  os << "void LLVM_ATTRIBUTE_UNUSED populateWithGenerated(mlir::MLIRContext "
        "*context, mlir::OwningRewritePatternList *patterns) {\n";
  for (const auto &name : tacticsNames) {
    os << "  patterns->insert<" << name << ">(context);\n";
  }
  os << "}\n";
}

static mlir::GenRegistration genMatchers("gen-tactics-defs", "Generate tactics",
                                         [](const RecordKeeper &records,
                                            raw_ostream &os) {
                                           emitMatchersRewriters(records, os);
                                           return false;
                                         });
