#include "MatchersGen.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"

using namespace llvm;
using namespace mlir;
using namespace lang;

using mlir::tblgen::Operator;

#define DEBUG_TYPE "mlir-tblgen-tactics"

static llvm::cl::opt<bool> clEmitBlas("emit-blas",
                                      llvm::cl::desc("directly emit blas call"),
                                      llvm::cl::init(false));

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

std::vector<StringRef> BuilderEmitter::getField(StringRef id) {
  auto record = record_->getValueAsDef(id);
  if (id.equals("affineExpr")) {
    std::vector<StringRef> res = {record->getValueAsString(id)};
    return res;
  }
  return record->getValueAsListOfStrings(id);
}

void BuilderEmitter::emitMatmulHelpers(std::string A, std::string B,
                                       std::string C) {
  os << formatv(
      R"(
    auto getOperandFromParamsMatmul = [&]() {
        llvm::SmallVector<mlir::Value, 3> res = { {0}, {1}, {2} };
        return res;
    };)",
      C, A, B);
}

// TODO: check how to properly escape { and }
// https://llvm.org/doxygen/FormatVariadic_8h_source.html
// mlir::Type{{} does not make a lot of sense.
void BuilderEmitter::emitMatmulBlas(std::string A, std::string B,
                                    std::string C) {
  os << formatv(
      R"(
    auto module = op.getParentOfType<mlir::ModuleOp>();
    auto fn = composeFunctionCallName(FUNCTION::MATMUL,
      llvm::ArrayRef<mlir::Type>{ {0}.getType(), {1}.getType(), {2}.getType() });
    auto symbolFn = getOrInsertFunction(rewriter, module, fn,
      llvm::ArrayRef<mlir::Type>{ {0}.getType(), {1}.getType(), {2}.getType() });
    rewriter.create<mlir::CallOp>(op.getLoc(), symbolFn, llvm::ArrayRef<mlir::Type>{{},
      llvm::ArrayRef<mlir::Value>{ {0}, {1}, {2} }); 
    )",
      C, A, B);
}

void BuilderEmitter::emitMatvecBlas(std::string A, std::string x,
                                    std::string y) {
  os << formatv(
      R"(
    auto module = op.getParentOfType<mlir::ModuleOp>();
    auto fn = composeFunctionCallName(FUNCTION::MATVEC,
      llvm::ArrayRef<mlir::Type>{ {0}.getType(), {1}.getType(), {2}.getType() });
    auto symbolFn = getOrInsertFunction(rewriter, module, fn,
      llvm::ArrayRef<mlir::Type>{ {0}.getType(), {1}.getType(), {2}.getType() });
    rewriter.create<mlir::CallOp>(op.getLoc(), symbolFn, llvm::ArrayRef<mlir::Type>{{},
      llvm::ArrayRef<mlir::Value>{ {0}, {1}, {2} }); 
    )",
      x, A, y);
}

SmallVector<std::string, 3> BuilderEmitter::getInputOperands() {
  auto inputs = getField("inputs");
  SmallVector<std::string, 3> lookupOperands;
  std::string lookupName;
  for (const auto &input : inputs) {
    if (!symbolTable_.lookup(input.str(), lookupName))
      llvm_unreachable("cannot find symbol");
    lookupOperands.push_back(lookupName);
  }
  return lookupOperands;
}

void BuilderEmitter::emitMatvec(bool isEmitted, std::string destBuff) {
  assert((isEmitted == false) &&
         "matvec must not create a new buffer - in-place operation");
  auto lookupInputOperands = getInputOperands();
  assert((lookupInputOperands.size() == 2) && "expect 2 args for matvec");
  if (!clEmitBlas)
    os << "assert(0);\n";
  else
    emitMatvecBlas(lookupInputOperands[0], lookupInputOperands[1], destBuff);
}

void BuilderEmitter::emitMatmul(bool isEmitted, std::string destBuff) {
  assert((isEmitted == false) &&
         "matmul must not emit a new buffer - in-place computation");
  auto lookupInputOperands = getInputOperands();
  assert((lookupInputOperands.size() == 2) && "expect 2 args for matmul");
  if (!clEmitBlas) {
    emitMatmulHelpers(lookupInputOperands[0], lookupInputOperands[1], destBuff);
    os << record_->getValueAsString("body");
  } else
    emitMatmulBlas(lookupInputOperands[0], lookupInputOperands[1], destBuff);
}

void BuilderEmitter::emitTransposeHelpers() {
  auto inputs = getField("inputs");
  auto affineExpr = getField("affineExpr");

  assert((affineExpr.size() == 1) && "expect single affine expr for transpose");
  assert((inputs.size() == 1) && "expect single input for transpose");

  std::string lookupInput;
  if (!symbolTable_.lookup(inputs[0].str(), lookupInput))
    llvm_unreachable("cannot find symbol");

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
      affineExpr[0]);
}

void BuilderEmitter::emitTransposeBlas(bool isEmitted, std::string destBuff) {
  auto lookupInputOperand = getInputOperands();
  assert((lookupInputOperand.size() == 1) && "expect single operand");
  auto permutations = getField("affineExpr");
  assert((permutations.size()) == 1 &&
         "expect single permutation for transpose");
  auto permutation = permutations[0];
  // each transpose operation does the following:
  // 1. create a new memref, if "isEmitted" is assert.
  // 2. compose the function call and create a global
  // array of ints to represent the permuation.
  // 3. create the signature for the transpose fn call.
  // (memref source, memref dest, int* perm, int perm.size()).
  os << formatv(
      R"(
    auto module = op.getParentOfType<mlir::ModuleOp>();
  )");
  if (!isEmitted) {
    os << formatv(
        R"(
    auto tType = {0}.getType().dyn_cast<mlir::MemRefType>();
    )",
        destBuff);
  } else {
    os << formatv(
        R"(
    auto tType = getTransposedMemref(
      {0}.getType().dyn_cast<mlir::MemRefType>(), {1});
    {2} = rewriter.create<mlir::AllocOp>(op.getLoc(), tType).getResult();
    )",
        lookupInputOperand[0], permutation, destBuff);
  }
  os << formatv(
      R"(
    auto fn = composeFunctionCallName(FUNCTION::TRANSPOSE,
      llvm::ArrayRef<mlir::Type>{ {0}.getType(), tType });
    auto *llvmDialect = op.getContext()->getRegisteredDialect<mlir::LLVM::LLVMDialect>();
    assert(llvmDialect && "expected llvm dialect to be registered");
    auto global = getOrCreateGlobalArray(op.getLoc(), rewriter, getPermutationArrayName( {1} ),
      {1}, module, llvmDialect);
    mlir::Value permutationSize = 
      getPermutationSizeAsConstantOp(op.getLoc(), rewriter, {1}, llvmDialect ); 
    auto symbolFn = getOrInsertFunction(rewriter, module, fn,
      llvm::ArrayRef<mlir::Type>{ {0}.getType(), tType, global.getType(), permutationSize.getType() });
    rewriter.create<mlir::CallOp>(op.getLoc(), symbolFn, llvm::ArrayRef<mlir::Type>{{},
      llvm::ArrayRef<mlir::Value>{ {0}, {2}, global, permutationSize });
  )",
      lookupInputOperand[0], permutation, destBuff);
}

void BuilderEmitter::emitTranspose(bool isEmitted, std::string destBuff) {
  if (!clEmitBlas) {
    emitTransposeHelpers();
    os << record_->getValueAsString("body");
    os << formatv(
        R"(
    auto permutationMap = mlir::AffineMap::getPermutationMap(
      getPermutationMapFromParamsPermute(), rewriter.getContext());
    {0} = rewriter.create<mlir::linalg::TransposeOp>(
      op.getLoc(), getOperandFromParamsPermute(), mlir::AffineMapAttr::get(permutationMap));
    )",
        destBuff);
  } else
    emitTransposeBlas(isEmitted, destBuff);
}

void BuilderEmitter::emitReshapeBlas(bool isEmitted, std::string destBuff) {
  auto lookupInputOperand = getInputOperands();
  assert((lookupInputOperand.size() == 1) && "expect single operand");
  auto indexMaps = getField("affineExpr");
  auto indexMap = indexMaps[0];

  os << formatv(
      R"(
    auto module = op.getParentOfType<mlir::ModuleOp>();
  )");
  if (!isEmitted) {
    os << formatv(
        R"(
    auto tType = {0}.getType().dyn_cast<mlir::MemRefType>();
    )",
        destBuff);
  } else {
    os << formatv(
        R"(
    auto tType = getReshapedMemRef(
      {0}.getType().dyn_cast<mlir::MemRefType>(), {1});
    {2} = rewriter.create<mlir::AllocOp>(op.getLoc(), tType).getResult();
  )",
        lookupInputOperand[0], indexMap, destBuff);
  }
  os << formatv(
      R"(
    auto fn = composeFunctionCallName(FUNCTION::RESHAPE,
      llvm::ArrayRef<mlir::Type>{ {0}.getType(), tType});
    auto symbolFn = getOrInsertFunction(rewriter, module, fn,
      llvm::ArrayRef<mlir::Type>{ {0}.getType(), tType});
    rewriter.create<mlir::CallOp>(op.getLoc(), symbolFn, llvm::ArrayRef<mlir::Type>{{},
      llvm::ArrayRef<mlir::Value>{ {0}, {1} });
  )",
      lookupInputOperand[0], destBuff);
}

void BuilderEmitter::emitReshape(bool isEmitted, std::string destBuff) {
  if (!clEmitBlas)
    os << "assert(0);\n";
  else
    emitReshapeBlas(isEmitted, destBuff);
}

void BuilderEmitter::emitErase() { os << record_->getValueAsString("body"); }

// The preamble looks like:
//  ```
//  mir::Value emittedVar
//  { // scope builder name
//  ```
// emittedVar is emitted only if the output
// is not already available in the symbol table.
// We also insert the new variable in the symbol table and
// bind it with the output.
void BuilderEmitter::emitPreamble(bool &isEmitted, std::string &dest) {
  // get new variable name if output not defined yet.
  auto outputs = getField("outputs");
  assert((outputs.size() == 1) && "expect single output");
  isEmitted = false;
  dest = outputs[0].str();
  std::string lookupSymbol;
  if (symbolTable_.lookup(dest, lookupSymbol)) {
    dest = lookupSymbol;
  } else {
    auto emittedVar = symbolTable_.getNextVariable();
    isEmitted = true;
    dest = emittedVar;
    symbolTable_.updateOrInsert(outputs[0].str(), emittedVar);
    os << formatv(
        R"(
    mlir::Value {0};
    )",
        emittedVar);
  }
  os << formatv(
      R"(
    { // start scope {0}
  )",
      record_->getValueAsString("name"));
  os << "\n";
}

// The postamble looks like:
//
// ```
// } // scope builder name
// ```
void BuilderEmitter::emitPostamble() {
  os << "\n";
  os << formatv(
      R"(
    } // end scope {0}
  )",
      record_->getValueAsString("name"));
  os << "\n\n";
}

void BuilderEmitter::emit() {
  auto builderName = record_->getValueAsString("name");
  LLVM_DEBUG(dbgs() << "emitting ---> " << builderName << "\n");
  std::string dest = "unknown";
  bool isEmitted = false;
  if (builderName.equals("matmul")) {
    emitPreamble(isEmitted, dest);
    emitMatmul(isEmitted, dest);
    emitPostamble();
  }
  if (builderName.equals("matvec")) {
    emitPreamble(isEmitted, dest);
    emitMatvec(isEmitted, dest);
    emitPostamble();
  }
  if (builderName.equals("permute")) {
    emitPreamble(isEmitted, dest);
    emitTranspose(isEmitted, dest);
    emitPostamble();
  }
  if (builderName.equals("reshape")) {
    emitPreamble(isEmitted, dest);
    emitReshape(isEmitted, dest);
    emitPostamble();
  }
  if (builderName.equals("erase"))
    emitErase();
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

bool SymbolTableMap::lookup(std::string key, std::string &value) const {
  auto it = symbolTable_.find(key);
  if (it != symbolTable_.end()) {
    value = it->second;
    return true;
  }
  return false;
}

bool SymbolTableMap::lookup(std::string key) const {
  std::string dummyCapture;
  return lookup(key, dummyCapture);
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
  emitArithOperationMatcher(t->trees().at(0));
  os << ", ";
  emitArithOperationMatcher(t->trees().at(1));
  os << ")";
}

void TacticsEmitter::emitConstantOperationMatcher(const Const &cst) {
  assert(0 && "not implemented");
}

void TacticsEmitter::emitArithOperationMatcher(const TreeRef &t) {
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
    std::string lookupName;
    if (!symbolTable_.lookup(tcName.name(), lookupName))
      llvm_unreachable("cannot find symbol");
    os << lookupName;
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
    os << "m_Op<mlir::AddFOp>(";
    std::string lookupName;
    if (!symbolTable_.lookup(comprehension.ident().name(), lookupName))
      llvm_unreachable("cannot find symbol");
    os << lookupName << ", ";
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
        return mlir::failure();
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
  mlir::LogicalResult matchAndRewrite(mlir::AffineForOp op,
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
  os.indent(4) << "return mlir::success();\n";
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

static mlir::GenRegistration
    genMatchersLinalg("gen-tactics-linalg", "Generate tactics for linalg",
                      [](const RecordKeeper &records, raw_ostream &os) {
                        emitMatchersRewriters(records, os);
                        return false;
                      });
static mlir::GenRegistration
    genMatchersBlas("gen-tactics-blas", "Generate tactics for blas calls",
                    [](const RecordKeeper &records, raw_ostream &os) {
                      clEmitBlas = true;
                      emitMatchersRewriters(records, os);
                      return false;
                    });
