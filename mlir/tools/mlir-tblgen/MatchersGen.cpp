#include "MatchersGen.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"

using namespace llvm;
using namespace mlir;
using namespace lang;

using mlir::tblgen::Operator;

#define DEBUG_TYPE "mlir-tblgen-tactics"

static llvm::cl::opt<bool>
    clEmitBlasCpu("emit-blas-cpu",
                  llvm::cl::desc("directly emit blas call for cpu"),
                  llvm::cl::init(false));
static llvm::cl::opt<bool>
    clEmitBlasGpu("emit-blas-gpu",
                  llvm::cl::desc("directly emit blas call for gpu"),
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

thread_local SymbolTableMap<std::string> BuilderEmitter::symbolTable_;

BuilderEmitter::BuilderEmitter(Record *record, bool lastBeforeEraseOp,
                               raw_ostream &os)
    : record_(record), lastBeforeEraseOp_(lastBeforeEraseOp), os(os){};

// TODO: make name consistent map and permuation.
StringRef ReshapeBlasEntry::getMap() const {
  auto record = record_->getValueAsDef("affineExpr");
  return record->getValueAsString("stringExpr");
}

StringRef ReshapeBlasEntry::getInputs() const {
  auto record = record_->getValueAsDef("inputs");
  auto res = record->getValueAsListOfStrings("inputs");
  assert(res.size() == 1 && "expect single input for reshape");
  return res[0];
}

StringRef ReshapeBlasEntry::getOutputs() const {
  auto record = record_->getValueAsDef("outputs");
  auto res = record->getValueAsListOfStrings("outputs");
  assert(res.size() == 1 && "expect single output for reshape");
  return res[0];
}

// TODO: make name consistent affineExpr and permuation.
StringRef TransposeBlasEntry::getPermutation() const {
  auto record = record_->getValueAsDef("affineExpr");
  return record->getValueAsString("stringExpr");
}

StringRef TransposeBlasEntry::getInputs() const {
  auto record = record_->getValueAsDef("inputs");
  auto res = record->getValueAsListOfStrings("inputs");
  assert(res.size() == 1 && "expect single input for transpose");
  return res[0];
}

StringRef TransposeBlasEntry::getOutputs() const {
  auto record = record_->getValueAsDef("outputs");
  auto res = record->getValueAsListOfStrings("outputs");
  assert(res.size() == 1 && "expect single output for transpose");
  return res[0];
}

StringRef MatvecBlasEntry::getAlpha() const {
  auto record = record_->getValueAsDef("alpha");
  return record->getValueAsString("valueConstant");
}

StringRef MatvecBlasEntry::getBeta() const {
  auto record = record_->getValueAsDef("beta");
  return record->getValueAsString("valueConstant");
}

StringRef MatvecBlasEntry::getTransA() const {
  auto record = record_->getValueAsDef("transA");
  return record->getValueAsString("stringExpr");
}

std::vector<llvm::StringRef> MatvecBlasEntry::getInputs() const {
  auto record = record_->getValueAsDef("inputs");
  auto res = record->getValueAsListOfStrings("inputs");
  assert(res.size() == 2 && "expect two inputs for matmul");
  return res;
}

StringRef MatvecBlasEntry::getOutputs() const {
  auto record = record_->getValueAsDef("outputs");
  auto res = record->getValueAsListOfStrings("outputs");
  assert(res.size() == 1 && "expect one output for matmul");
  return res[0];
}

StringRef MatmulBlasEntry::getAlpha() const {
  auto record = record_->getValueAsDef("alpha");
  return record->getValueAsString("valueConstant");
}

StringRef MatmulBlasEntry::getBeta() const {
  auto record = record_->getValueAsDef("beta");
  return record->getValueAsString("valueConstant");
}

int64_t MatmulBlasEntry::getDimensionForM() const {
  auto record = record_->getValueAsDef("m");
  return record->getValueAsInt("value");
}

int64_t MatmulBlasEntry::getDimensionForN() const {
  auto record = record_->getValueAsDef("n");
  return record->getValueAsInt("value");
}

int64_t MatmulBlasEntry::getDimensionForK() const {
  auto record = record_->getValueAsDef("k");
  return record->getValueAsInt("value");
}

StringRef MatmulBlasEntry::getTransA() const {
  auto record = record_->getValueAsDef("transA");
  return record->getValueAsString("stringExpr");
}

StringRef MatmulBlasEntry::getTransB() const {
  auto record = record_->getValueAsDef("transB");
  return record->getValueAsString("stringExpr");
}

std::vector<llvm::StringRef> MatmulBlasEntry::getInputs() const {
  auto record = record_->getValueAsDef("inputs");
  auto res = record->getValueAsListOfStrings("inputs");
  assert(res.size() == 2 && "expect two inputs for matmul");
  return res;
}

StringRef MatmulBlasEntry::getOutputs() const {
  auto record = record_->getValueAsDef("outputs");
  auto res = record->getValueAsListOfStrings("outputs");
  assert(res.size() == 1 && "expect one output for matmul");
  return res[0];
}

std::vector<llvm::StringRef> ConvBlasEntry::getInputs() const {
  auto record = record_->getValueAsDef("inputs");
  auto res = record->getValueAsListOfStrings("inputs");
  assert(res.size() == 2 && "expect two inputs for conv");
  return res;
}

StringRef ConvBlasEntry::getOutputs() const {
  auto record = record_->getValueAsDef("outputs");
  auto res = record->getValueAsListOfStrings("outputs");
  assert(res.size() == 1 && "expect one output for conv");
  return res[0];
}

// TODO: remove duplicate code.
StringRef ConvBlasEntry::getPadding() const {
  auto record = record_->getValueAsDef("padding");
  auto res = record->getValueAsString("stringExpr");
  return res;
}

StringRef ConvBlasEntry::getStride() const {
  auto record = record_->getValueAsDef("stride");
  auto res = record->getValueAsString("stringExpr");
  return res;
}

bool isConstantOne(const std::string &s) {
  if (s.empty())
    return false;
  auto isNumber = std::find_if(s.begin(), s.end(), [](unsigned char c) {
                    return !std::isdigit(c);
                  }) == s.end();
  if (!isNumber)
    return false;
  int number = std::stoi(s);
  return number == 1;
}

void BuilderEmitter::emitMatmulLinalg(MatmulTy &mmi) {
  auto C = mmi.output;
  auto A = mmi.inputs[0];
  auto B = mmi.inputs[1];
  auto alpha = mmi.alpha;
  auto beta = mmi.beta;

  os << formatv(
      R"(
    createLinalgMatmulOp(rewriter, op.getLoc(), {0}, {1}, {2}, {3}, {4});
    )",
      beta, alpha, C, A, B);
}

void BuilderEmitter::emitMatmulBlas(MatmulTy &mmi, Target t) {

  auto C = mmi.output;
  auto A = mmi.inputs[0];
  auto B = mmi.inputs[1];

  switch (t) {
  case Target::CPU: {
    os << formatv(
        R"(
    auto module = op.getParentOfType<mlir::ModuleOp>();
    createCallToMklSgemm(module, rewriter, op.getLoc(), {0}, {1}, {2}, {3}, {4}, {5}, {6}, {7}, {8}, {9});
    )",
        C, A, B, mmi.alpha, mmi.beta, mmi.transA, mmi.transB, mmi.dimForM,
        mmi.dimForN, mmi.dimForK);
    return;
  }
  // TODO: handle alpha and beta for GPU.
  case Target::GPU: {
    os << formatv(
        R"(
    auto module = op.getParentOfType<mlir::ModuleOp>();
    auto devC = createCallAllocateMemoryForDevice(module, rewriter, op.getLoc(), getSizeBuffer({0}.getType()));
    auto devA = createCallAllocateMemoryForDevice(module, rewriter, op.getLoc(), getSizeBuffer({1}.getType()));
    auto devB = createCallAllocateMemoryForDevice(module, rewriter, op.getLoc(), getSizeBuffer({2}.getType())); 
    createCallCopyFromHostToDevice(module, rewriter, op.getLoc(), {0}, devC, 
                                   getSizeBuffer({0}.getType()));
    createCallCopyFromHostToDevice(module, rewriter, op.getLoc(), {1}, devA, 
                                   getSizeBuffer({1}.getType()));
    createCallCopyFromHostToDevice(module, rewriter, op.getLoc(), {2}, devB, 
                                   getSizeBuffer({2}.getType()));
    createCallToCublasSgemm(module, rewriter, op.getLoc(), devC, devA, devB, {0}, {1}, {2});
    createCallCopyFromDeviceToHost(module, rewriter, op.getLoc(), devC, {0}, 
                                   getSizeBuffer({0}.getType()));    
    )",
        C, A, B);
    return;
  }
    assert(0 && "case not supported for emitMatmulBlas");
  }
}

void BuilderEmitter::emitMatmul(bool isEmitted, std::string destBuff) {
  assert((isEmitted == false) &&
         "matmul must not emit a new buffer - in-place computation");
  auto matmulEntry = MatmulBlasEntry(record_);
  MatmulTy mmi;
  mmi.alpha = matmulEntry.getAlpha().str();
  mmi.beta = matmulEntry.getBeta().str();
  mmi.dimForM = matmulEntry.getDimensionForM();
  mmi.dimForN = matmulEntry.getDimensionForN();
  mmi.dimForK = matmulEntry.getDimensionForK();
  mmi.transA = (matmulEntry.getTransA() == "N") ? false : true;
  mmi.transB = (matmulEntry.getTransB() == "N") ? false : true;
  mmi.inputs = lookUpOperands(matmulEntry.getInputs());
  mmi.output = destBuff;

  if (clEmitBlasGpu) {
    emitMatmulBlas(mmi, Target::GPU);
    return;
  }
  if (clEmitBlasCpu) {
    emitMatmulBlas(mmi, Target::CPU);
    return;
  }
  emitMatmulLinalg(mmi);
}

void BuilderEmitter::emitConvolution(bool isEmitted, std::string destBuff) {
  // this can be relaxed but we need to emit a new buffer.
  assert((isEmitted == false) &&
         "convolution must not emit a new buffer - in-place computation");
  auto convEntry = ConvBlasEntry(record_);
  auto inputs = lookUpOperands(convEntry.getInputs());
  auto output = destBuff;
  auto padding = convEntry.getPadding();
  auto stride = convEntry.getStride();
  if (clEmitBlasCpu) {
    os << formatv(
        R"(
    auto module = op.getParentOfType<mlir::ModuleOp>();
    createCallToMklConvolution(module, rewriter, op.getLoc(), {0}, {1}, {2}, {3}, {4});
    )",
        inputs[0], inputs[1], output, padding, stride);
  }
}

void BuilderEmitter::emitMatvecBlas(MatvecTy &mvi) {
  auto x = mvi.output;
  auto A = mvi.inputs[0];
  auto y = mvi.inputs[1];

  os << formatv(
      R"(
    auto module = op.getParentOfType<mlir::ModuleOp>();
    createCallToMklSgemv(module, rewriter, op.getLoc(), {0}, {1}, {2}, {3}, {4}, {5});
    )",
      x, A, y, mvi.alpha, mvi.beta, mvi.transA);
}

void BuilderEmitter::emitMatvecLinalg(MatvecTy &mvi) {
  auto x = mvi.output;
  auto A = mvi.inputs[0];
  auto y = mvi.inputs[1];
  auto alpha = mvi.alpha;
  auto beta = mvi.beta;

  os << formatv(
      R"(
    createLinalgMatvecOp(rewriter, op.getLoc(), {0}, {1}, {2}, {3}, {4});
    )",
      beta, alpha, A, y, x);
}

void BuilderEmitter::emitMatvec(bool isEmitted, std::string destBuff) {
  assert((isEmitted == false) &&
         "matvec must not create a new buffer - in-place operation");
  auto matvecEntry = MatvecBlasEntry(record_);
  MatvecTy mvi;
  mvi.alpha = matvecEntry.getAlpha().str();
  mvi.beta = matvecEntry.getBeta().str();
  mvi.transA = (matvecEntry.getTransA() == "N") ? false : true;
  mvi.inputs = lookUpOperands(matvecEntry.getInputs());
  mvi.output = destBuff;
  if (clEmitBlasCpu)
    emitMatvecBlas(mvi);
  else
    emitMatvecLinalg(mvi);
}

std::string BuilderEmitter::lookUpOperand(StringRef operand) const {
  std::string lookupOperand;
  if (!symbolTable_.lookup(operand.str(), lookupOperand))
    llvm_unreachable("cannot find symbol");
  return lookupOperand;
}

std::vector<std::string>
BuilderEmitter::lookUpOperands(std::vector<StringRef> operands) const {
  std::vector<std::string> lookupOperands;
  for (const auto &operand : operands) {
    std::string lookupOperand;
    if (!symbolTable_.lookup(operand.str(), lookupOperand))
      llvm_unreachable("cannot find symbol");
    lookupOperands.push_back(lookupOperand);
  }
  return lookupOperands;
}

void BuilderEmitter::emitTransposeBlas(bool isEmitted, std::string destBuff,
                                       std::string input,
                                       std::string permutation) {
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

  if (isEmitted) {
    os << formatv(
        R"(
    auto tType = getTransposedMemref(
      {0}.getType().dyn_cast<mlir::MemRefType>(), {1});
    {2} = rewriter.create<mlir::AllocOp>(op.getLoc(), tType).getResult();
    )",
        input, permutation, destBuff);
  }
  os << formatv(
      R"(
    createCallToMklTranspose(module, rewriter, op.getLoc(), {0}, {2}, {1}); 
  )",
      input, permutation, destBuff);
}

void BuilderEmitter::emitTransposeLinalg(std::string destBuff,
                                         std::string input,
                                         std::string permutation) {
  os << formatv(
      R"(
    auto permutationMap = mlir::AffineMap::getPermutationMap(
      llvm::ArrayRef<unsigned>({1}), rewriter.getContext());
    
    // Emit a linalg transpose as:
    // %0 = memref_cast %arg0 : memref<32x1024x32xf32> to memref<?x?x?xf32>
    // %1 = linalg.transpose %0 (d0, d1, d2) -> (d2, d0, d1) : memref<?x?x?xf32>
    // %2 = memref_cast %1 : memref<?x?x?xf32> to memref<32x32x1024xf32>

    auto inputMemRef = {0}.getType().dyn_cast_or_null<mlir::MemRefType>();
    llvm::SmallVector<int64_t, 8> newSizes;
    size_t numDimInput = inputMemRef.getShape().size();
    newSizes.reserve(numDimInput);
    for (size_t i = 0; i < numDimInput; i++)
      newSizes.push_back(-1);
    mlir::MemRefType newMemRefType =
      mlir::MemRefType::Builder(inputMemRef).setShape(newSizes).setAffineMaps({{});
    
    mlir::Value casted = 
      rewriter.create<mlir::MemRefCastOp>(op.getLoc(), newMemRefType, {0});

    mlir::Value transposed = rewriter.create<mlir::linalg::TransposeOp>(
      op.getLoc(), casted, mlir::AffineMapAttr::get(permutationMap));
    
    newSizes = applyPermutation(inputMemRef.getShape(), {1}); 
    // XXX
    transposed.setType(newMemRefType);
    //auto transposedType = transposed.getType().dyn_cast_or_null<mlir::MemRefType>();
    newMemRefType = mlir::MemRefType::Builder(inputMemRef)
      .setShape(newSizes)/*.setAffineMaps(transposedType.getAffineMaps())*/;
    )",
      input, permutation);

  if (lastBeforeEraseOp_) {
    os << formatv(
        R"(
    mlir::Value castedBack = 
      rewriter.create<mlir::MemRefCastOp>(op.getLoc(), newMemRefType, transposed);
    rewriter.create<mlir::linalg::CopyOp>(op.getLoc(), castedBack, {0});
    )",
        destBuff);
  } else {
    os << formatv(
        R"( 
    {0} = rewriter.create<mlir::MemRefCastOp>(op.getLoc(), newMemRefType, transposed);
    )",
        destBuff);
  }
}

void BuilderEmitter::emitTranspose(bool isEmitted, std::string destBuff) {
  auto transposeBlasEntry = TransposeBlasEntry(record_);
  auto input = transposeBlasEntry.getInputs();
  auto lookupOperand = lookUpOperand(input);
  auto permutation = transposeBlasEntry.getPermutation();

  if (!clEmitBlasCpu)
    emitTransposeLinalg(destBuff, lookupOperand, permutation.str());
  else
    emitTransposeBlas(isEmitted, destBuff, lookupOperand, permutation.str());
}

void BuilderEmitter::emitReshapeBlas(bool isEmitted, std::string destBuff,
                                     std::string input, std::string indexMap) {
  os << formatv(
      R"(
    auto module = op.getParentOfType<mlir::ModuleOp>();
  )");

  if (isEmitted) {
    os << formatv(
        R"(
    auto tType = getReshapedMemRef(
      {0}.getType().dyn_cast<mlir::MemRefType>(), {1});
    {2} = rewriter.create<mlir::AllocOp>(op.getLoc(), tType).getResult();
  )",
        input, indexMap, destBuff);
  }
  os << formatv(
      R"(
    createCallToMklReshape(module, rewriter, op.getLoc(), {0}, {1}); 
  )",
      input, destBuff);
}

void BuilderEmitter::emitReshapeLinalg(std::string destBuff, std::string input,
                                       std::string indexMap) {
  if (lastBeforeEraseOp_) {
    os << formatv(
        R"(
    mlir::Value r = createLinalgReshapeOp(rewriter, op.getLoc(), {1}, {2}, {0});
    rewriter.create<mlir::linalg::CopyOp>(op.getLoc(), r, {0});
    )",
        destBuff, input, indexMap);
  } else {
    os << formatv(
        R"(
    {0} = createLinalgReshapeOp(rewriter, op.getLoc(), {1}, {2}, {0});
      )",
        destBuff, input, indexMap);
  }
}

void BuilderEmitter::emitReshape(bool isEmitted, std::string destBuff) {
  auto reshapeBlasEntry = ReshapeBlasEntry(record_);
  auto input = reshapeBlasEntry.getInputs();
  auto lookupOperand = lookUpOperand(input);
  auto indexMap = reshapeBlasEntry.getMap();

  if (!clEmitBlasCpu)
    emitReshapeLinalg(destBuff, lookupOperand, indexMap.str());
  else
    emitReshapeBlas(isEmitted, destBuff, lookupOperand, indexMap.str());
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
void BuilderEmitter::emitPreamble(bool &isEmitted, std::string &dest,
                                  StringRef output) {
  isEmitted = false;
  dest = output.str();
  std::string lookupSymbol;
  if (symbolTable_.lookup(dest, lookupSymbol)) {
    dest = lookupSymbol;
  } else {
    auto emittedVar = symbolTable_.getNextVariable();
    isEmitted = true;
    dest = emittedVar;
    symbolTable_.updateOrInsert(output.str(), emittedVar);
    os << formatv(
        R"(
    mlir::Value {0} = nullptr;
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
    emitPreamble(isEmitted, dest, MatmulBlasEntry(record_).getOutputs());
    emitMatmul(isEmitted, dest);
    emitPostamble();
  }
  if (builderName.equals("matvec")) {
    emitPreamble(isEmitted, dest, MatvecBlasEntry(record_).getOutputs());
    emitMatvec(isEmitted, dest);
    emitPostamble();
  }
  if (builderName.equals("transpose")) {
    emitPreamble(isEmitted, dest, TransposeBlasEntry(record_).getOutputs());
    emitTranspose(isEmitted, dest);
    emitPostamble();
  }
  if (builderName.equals("reshape")) {
    emitPreamble(isEmitted, dest, ReshapeBlasEntry(record_).getOutputs());
    emitReshape(isEmitted, dest);
    emitPostamble();
  }
  if (builderName.equals("conv")) {
    emitPreamble(isEmitted, dest, ConvBlasEntry(record_).getOutputs());
    emitConvolution(isEmitted, dest);
    emitPostamble();
  }
  if (builderName.equals("erase"))
    emitErase();
}

template <> void SymbolTableMap<std::string>::dump() const {
  for (const auto &elem : symbolTable_) {
    LLVM_DEBUG(dbgs() << "elem.first  : " << elem.first << "\n");
    LLVM_DEBUG(dbgs() << "elem.second : " << elem.second << "\n");
    LLVM_DEBUG(dbgs() << "-----\n");
  }
}

template <> void SymbolTableMap<Tensor>::dump() const {
  for (const auto &elem : symbolTable_) {
    LLVM_DEBUG(dbgs() << "tensor.id : " << elem.first.getId() << " ");
    LLVM_DEBUG(dbgs() << "[ ");
    auto indexes = elem.first.getIndexes();
    for (const auto &index : indexes)
      LLVM_DEBUG(dbgs() << index.getId() << " ");
    LLVM_DEBUG(dbgs() << "]\n");
    LLVM_DEBUG(dbgs() << elem.second << "\n");
    LLVM_DEBUG(dbgs() << "-----\n");
  }
}

// collect iterators from comprehension.
using identifierInductions = SmallSet<std::string, 8>;
using identifierTensors = SmallSet<std::pair<bool, std::string>, 8>;
std::pair<identifierInductions, identifierTensors>
collectIteratorsAndTensorNames(const Comprehension &comprehension) {
  SmallSet<std::string, 8> iterators;
  SmallSet<std::pair<bool, std::string>, 8> tensors;

  for (const auto &lhs : comprehension.indices()) {
    iterators.insert(lhs.name());
  }
  tensors.insert(std::make_pair(true, comprehension.ident().name()));

  walkTree(comprehension.rhs(), [&](const TreeRef &t) {
    if (t->kind() == TK_APPLY) {
      auto tc = Apply(t);
      tensors.insert(std::make_pair(true, tc.name().name()));
      auto tcIters = tc.arguments();
      for (const auto &tcIter : tcIters) {
        if (tcIter->kind() == TK_IDENT)
          iterators.insert(Ident(tcIter).name());
      }
    }
    // detect scalar (i.e., alpha) which
    // are represented as tk_ident. Make
    // sure not to push iterators in the array name set.
    // mark the scalar such that we avoid emitting m_ArrayPlaceholder.
    if (t->kind() == TK_IDENT) {
      auto tcName = Ident(t).name();
      if (std::find(iterators.begin(), iterators.end(), tcName) ==
          iterators.end()) {
        auto it = std::find_if(
            tensors.begin(), tensors.end(),
            [&](std::pair<bool, std::string> p) { return p.second == tcName; });
        if (it == tensors.end())
          tensors.insert(std::make_pair(false, tcName));
      }
    }
  });
  return std::make_pair(iterators, tensors);
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
    return emitBinaryOperationMatcher(t, "mlir::DivFOp");
  case TK_NUMBER:
  case TK_CONST:
    return emitConstantOperationMatcher(Const(t));
  case TK_APPLY: {
    auto tc = Apply(t);
    std::string lookupName;
    if (!symbolTable_.lookup(Tensor::buildTensor(tc.name(), tc.arguments()),
                             lookupName))
      llvm_unreachable("cannot find symbol");
    os << lookupName;
    return;
  }
  case TK_IDENT: {
    auto scalar = Ident(t);
    os << "m_AnyCapture(" << scalar.name() << ")";
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
    if (!symbolTable_.lookup(
            Tensor::buildTensor(comprehension.ident(), comprehension.indices()),
            lookupName))
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

using identifierIterators = SmallSet<std::string, 8>;
using identifierTensors = SmallSet<std::pair<bool, std::string>, 8>;
void TacticsEmitter::emitAccessMatchLogic(
    const Comprehension &comprehension,
    const std::pair<identifierIterators, identifierTensors> &ids) {
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
    // do not emit m_ArrayPlaceholder if
    // we are dealing with a scalar.
    if (!id.first)
      continue;
    os.indent(8) << "auto ";
    os << "_" << id.second << " = m_ArrayPlaceholder();\n";
  }
  os << "\n";
  emitOperationMatchLogic(comprehension);
  // bind captured values.
  for (const auto &iterator : ids.first)
    os.indent(8) << iterator << " = "
                 << "pctx["
                 << "_" << iterator << "];\n";
  for (const auto &tensorName : ids.second) {
    if (!tensorName.first)
      continue;
    os.indent(8) << tensorName.second << " = "
                 << "pctx["
                 << "_" << tensorName.second << "];\n";
  }
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

using identifierIterators = SmallSet<std::string, 8>;
using identifierTensors = SmallSet<std::pair<bool, std::string>, 8>;
void TacticsEmitter::emitMatchLogic(
    const lang::Comprehension &comprehension,
    const std::pair<identifierIterators, identifierTensors>
        &iteratorsAndTensorNames) {

  // declare each iterators and array name as Value type. The
  // value will get filled if we match the pattern.
  auto iterators = iteratorsAndTensorNames.first;
  for (const auto &iterator : iterators)
    os.indent(4) << "mlir::Value " << iterator << ";"
                 << "\n";
  for (const auto &tensorName : iteratorsAndTensorNames.second)
    os.indent(4) << "mlir::Value " << tensorName.second << ";"
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
  size_t pos = 0;
  for (const auto builder : builders) {
    if (pos == builders.size() - 2)
      BuilderEmitter(builder, true, os).emit();
    else
      BuilderEmitter(builder, false, os).emit();
    pos++;
  }
}

TacticsEmitter::TacticsEmitter(Record *record, raw_ostream &os)
    : record_(record), loc_(record_->getLoc()),
      parser_(Parser(record_->getValueAsString("pattern").str())), os(os),
      symbolTable_(SymbolTableMap<Tensor>()){};

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
    BuilderEmitter::symbolTable_.insert(tensor.second, tensor.second);
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
    genMatchersBlasCpu("gen-tactics-blas-cpu",
                       "Generate tactics for blas calls on cpu",
                       [](const RecordKeeper &records, raw_ostream &os) {
                         clEmitBlasCpu = true;
                         emitMatchersRewriters(records, os);
                         return false;
                       });
static mlir::GenRegistration
    genMatchersBlasGpu("gen-tactics-blas-gpu",
                       "Generate tactics for blas calls in gpu",
                       [](const RecordKeeper &records, raw_ostream &os) {
                         clEmitBlasGpu = true;
                         emitMatchersRewriters(records, os);
                         return false;
                       });
