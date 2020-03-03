#include "MatchersGen.h"
#include "llvm/Support/Debug.h"

using namespace llvm;
using namespace mlir;
using namespace lang;

using mlir::tblgen::Operator;

#define DEBUG_TYPE "mlir-tblgen-tactics"

namespace llvm {
using identifierLine = std::pair<StringRef, unsigned>;
template <> struct format_provider<identifierLine> {
  static void format(const identifierLine &v, raw_ostream &os,
                     StringRef style) {
    os << v.first << ":" << v.second;
  }
};
} // end namespace llvm

thread_local SymbolTableMap BuilderEmitter::symbolTable_;

BuilderEmitter::BuilderEmitter(Record *record, raw_ostream &os)
    : record_(record), os(os){};

void BuilderEmitter::emitMatmulHelpers() {
  auto recordInput = record_->getValueAsDef("inputs");
  auto inputs = recordInput->getValueAsListOfStrings("inputs");

  auto recordOut = record_->getValueAsDef("outputs");
  auto outputs = recordOut->getValueAsListOfStrings("outputs");

  assert((outputs.size() == 1) && "expect single output for matmul");
  assert((inputs.size() > 1) && (inputs.size() < 4) &&
         "expect 2 or 3 inputs for matmul");

  std::vector<std::string> lookupInputs;
  for (const auto &input : inputs)
    lookupInputs.push_back(symbolTable_.lookup(input.str()));
  std::string lookupOutput = symbolTable_.lookup(outputs[0].str());

  // if inputs = 3, check that the output is also found as input.
  if (lookupInputs.size() == 3) {
    auto iterator =
        std::find(lookupInputs.begin(), lookupInputs.end(), lookupOutput);
    if (iterator == lookupInputs.end())
      assert(0 && "expect output to be found among the inputs");
    else
      lookupInputs.erase(iterator);
  }

  os << formatv(
      R"(
    auto getOperandFromParamsMatmul = [&]() {
        llvm::SmallVector<mlir::Value, 3> res = { {0}, {1}, {2} };
        return res;
    };)",
      lookupOutput, lookupInputs[0], lookupInputs[1]);
}

void BuilderEmitter::emitMatmul() {
  emitMatmulHelpers();
  os << record_->getValueAsString("body");
}

std::string BuilderEmitter::emitTransposeHelpers() {
  auto recordInput = record_->getValueAsDef("inputs");
  auto inputs = recordInput->getValueAsListOfStrings("inputs");

  auto recordOut = record_->getValueAsDef("outputs");
  auto outputs = recordOut->getValueAsListOfStrings("outputs");

  // maybe introduce some handy classes to operate on tablegen classes.
  auto affineExpr =
      record_->getValueAsDef("affineExpr")->getValueAsString("affineExpr");

  assert((outputs.size() == 1) && "expect single output for transpose");
  assert((inputs.size() == 1) && "expect single input for transpose");

  std::string lookupInput = symbolTable_.lookup(inputs[0].str());

  os << formatv(
      R"(
    auto getOperandFromParamsPermute = [&]() {
      return {0};
    };)",
      lookupInput);

  os << formatv(
      R"(
    auto getPermutationMapFromParamsPermute = []() {
      return llvm::ArrayRef<unsigned>({0});
    };)",
      affineExpr);

  return outputs[0].str();
}

std::string BuilderEmitter::emitTranspose() {
  auto resTranspose = emitTransposeHelpers();
  os << record_->getValueAsString("body");
  return resTranspose;
}

void BuilderEmitter::emitErase() { os << record_->getValueAsString("body"); }

void BuilderEmitter::emit() {
  auto builderName = record_->getValueAsString("name");
  LLVM_DEBUG(dbgs() << "emitting ---> " << builderName << "\n");

  if (builderName.equals("matmul")) {
    os.indent(4) << "{ // start scope matmul"
                 << "\n";
    emitMatmul();
    os << "\n";
    os.indent(4) << "} // end scope matmul"
                 << "\n\n";
    return;
  }
  if (builderName.equals("permute")) {
    auto emittedVar = symbolTable_.getNextVariable();
    os.indent(4) << "mlir::Value " << emittedVar << ";"
                 << "\n";
    os.indent(4) << "{ // start scope permute"
                 << "\n";
    auto resPermute = emitTranspose();
    os << "\n";
    os.indent(4) << emittedVar << " = permute; "
                 << "\n";
    os.indent(4) << "} // end scope matmul"
                 << "\n\n";
    symbolTable_.updateOrInsert(resPermute, emittedVar);
    return;
  }
  if (builderName.equals("erase")) {
    emitErase();
    return;
  }
  assert(0 && "case not convered");
}

void SymbolTableMap::dump() const {
  for (const auto &elem : symbolTable_) {
    LLVM_DEBUG(dbgs() << "elem.first  : " << elem.first << "\n");
    LLVM_DEBUG(dbgs() << "elem.second : " << elem.second << "\n");
    LLVM_DEBUG(dbgs() << "-----\n");
  }
}

void SymbolTableMap::clear() { symbolTable_.clear(); }

void SymbolTableMap::updateOrInsert(std::string key, std::string value) {
  auto it = symbolTable_.find(key);
  if (it == symbolTable_.end())
    insert(key, value);
  else
    it->second = value;
}

std::string SymbolTableMap::getNextVariable() {
  std::string var = "var" + std::to_string(nextId_++);
  return var;
}

void SymbolTableMap::insert(std::string key, std::string value) {
  symbolTable_.insert(std::make_pair(key, value));
}

std::string SymbolTableMap::lookup(std::string key) const {
  auto it = symbolTable_.find(key);
  if (it != symbolTable_.end())
    return it->second;
  LLVM_DEBUG(dbgs() << "key: " << key << "\n");
  dump();
  assert(0 && "expect value to be found");
  return "nullptr";
}

// Walk tree (see teckly).
void walkTree(const TreeRef &tree, std::function<void(const TreeRef &)> fn) {
  fn(tree);
  for (auto e : tree->trees())
    walkTree(e, fn);
}

// collect iterators from comprehension.
using identifier = SmallSet<std::string, 8>;
std::pair<identifier, identifier>
collectIteratorsAndTensorNames(const Comprehension &comprehension) {
  SmallSet<std::string, 8> iterators;
  SmallSet<std::string, 8> names;

  for (const auto &lhs : comprehension.indices()) {
    iterators.insert(lhs.name());
  }
  names.insert(comprehension.ident().name());

  walkTree(comprehension.rhs(), [&](const TreeRef &t) {
    if (t->kind() == TK_APPLY) {
      auto tc = Apply(t);
      names.insert(tc.name().name());
      auto tcIters = tc.arguments();
      for (const auto &tcIter : tcIters) {
        if (tcIter->kind() == TK_IDENT)
          iterators.insert(Ident(tcIter).name());
      }
    }
  });
  return std::make_pair(iterators, names);
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
    os << symbolTable_.lookup(tcName.name());
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

void TacticsEmitter::emitArithOperationMatcher(
    const lang::Comprehension &comprehension) {
  auto assignment = comprehension.assignment();
  auto rootOperation = assignment;
  switch (assignment->kind()) {
  case TK_PLUS_EQ: {
    os << "m_Op<mlir::AddFOp>("
       << symbolTable_.lookup(comprehension.ident().name()) << ", ";
    emitArithOperationMatcher(comprehension.rhs());
    os << ");";
    break;
  }
  case '=': {
    emitArithOperationMatcher(comprehension.rhs());
    os << ";";
    rootOperation = comprehension.rhs();
    break;
  }
  default:
    assert(0 && "assignment not supported");
  }

  auto rootOperationAsString = "undefined";
  switch (rootOperation->kind()) {
  case '*':
    rootOperationAsString = "mlir::MulFOp";
    break;
  case TK_PLUS_EQ:
    rootOperationAsString = "mlir::AddFOp";
    break;
  default:
    assert(0 && "root operation not supported");
  }

  os << formatv(
      R"(
        auto rootOp = 
          llvm::dyn_cast_or_null<{0}>(store.getValueToStore().getDefiningOp());
        if ((!rootOp) || (!bodyMatcher.match(rootOp)))
          return false;
    )",
      rootOperationAsString);
}

void TacticsEmitter::emitOperationMatchLogic(
    const Comprehension &comprehension) {
  // emit the only one store operation.
  auto assignment = comprehension.assignment();
  switch (assignment->kind()) {
  case '=':
    emitLoadOrStoreMatcherOp(comprehension.ident(), comprehension.indices(),
                             "mlir::AffineStoreOp");
    break;
  case TK_PLUS_EQ: {
    emitLoadOrStoreMatcherOp(comprehension.ident(), comprehension.indices(),
                             "mlir::AffineStoreOp");
    os << "\n";
    emitLoadOrStoreMatcherOp(comprehension.ident(), comprehension.indices(),
                             "mlir::AffineLoadOp");
    break;
  }
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
      emitLoadOrStoreMatcherOp(tc.name(), tc.arguments(), "mlir::AffineLoadOp");
      os << "\n";
    }
  });
  os.indent(8) << "auto bodyMatcher = ";
  // emit operation matcher.
  emitArithOperationMatcher(comprehension);
  os << "\n";
}

using identifier = SmallSet<std::string, 8>;
void TacticsEmitter::emitAccessMatchLogic(
    const Comprehension &comprehension,
    const std::pair<identifier, identifier> &ids) {
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
  // bind captured values.
  for (const auto &iterator : ids.first)
    os.indent(8) << iterator << " = "
                 << "pctx["
                 << "_" << iterator << "];\n";
  for (const auto &tensorName : ids.second)
    os.indent(8) << tensorName << " = "
                 << "pctx["
                 << "_" << tensorName << "];\n";
  os << R"(
        return true;
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

using identifier = llvm::SmallSet<std::string, 8>;
void TacticsEmitter::emitMatchLogic(
    const lang::Comprehension &comprehension,
    const std::pair<identifier, identifier> &iteratorsAndTensorNames) {

  // declare each iterators and array name as Value type. The
  // value will get filled if we match the pattern.
  auto iterators = iteratorsAndTensorNames.first;
  for (const auto &iterator : iterators)
    os.indent(4) << "mlir::Value " << iterator << ";"
                 << "\n";
  for (const auto &tensorName : iteratorsAndTensorNames.second)
    os.indent(4) << "mlir::Value " << tensorName << ";"
                 << "\n";
  os << R"(
    auto body = [&](mlir::Operation &op) -> bool {
      auto loop = llvm::cast<mlir::AffineForOp>(op);
      mlir::Region &loopBody = loop.getLoopBody();
      using namespace mlir::matchers;
  )";
  // emit placeholders and arrayPlaceholders.
  emitAccessMatchLogic(comprehension, iteratorsAndTensorNames);
  os << R"(
    }; // end body callback.
  )";
  // emit nestedMatchers.
  emitStructuralMatchLogic(iterators.size());
}

void TacticsEmitter::emitRewriteLogic() {
  auto builders = record_->getValueAsListOfDefs("builders");
  for (const auto builder : builders) {
    BuilderEmitter(builder, os).emit();
  }
}

TacticsEmitter::TacticsEmitter(Record *record, raw_ostream &os)
    : record_(record), loc_(record_->getLoc()),
      parser_(Parser(record_->getValueAsString("pattern").str())), os(os),
      symbolTable_(SymbolTableMap()){};

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
  os.indent(4) << "// Match\n";
  auto comprehension = Comprehension(parser_.parseStmt());
  auto iteratorsAndTensorNames = collectIteratorsAndTensorNames(comprehension);
  emitMatchLogic(comprehension, iteratorsAndTensorNames);
  os << "\n";
  os.indent(4) << "// Rewrite\n";
  // fill in the symbol table.
  for (const auto &tensor : iteratorsAndTensorNames.second)
    BuilderEmitter::symbolTable_.insert(tensor, tensor);
  emitRewriteLogic();
  BuilderEmitter::symbolTable_.clear();
  // erase the content when the tactics is done.
  os << "\n";
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
