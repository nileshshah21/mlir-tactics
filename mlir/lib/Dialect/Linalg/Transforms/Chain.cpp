#include "PassDetail.h"
#include "mlir/Dialect/Linalg/IR/LinalgOps.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "matmul-chain"

using namespace mlir;
using namespace mlir::linalg;

static SmallVector<long, 8> getPVector(const SmallVector<Value, 8> &values) {
  SmallVector<long, 8> pVector = {};
  for (const auto &value : values) {
    auto memRefType = value.getType().dyn_cast_or_null<MemRefType>();
    if (!memRefType)
      return {};
    auto shape = memRefType.getShape();
    if (!pVector.size()) {
      pVector.push_back(shape[0]);
      pVector.push_back(shape[1]);
    } else
      pVector.push_back(shape[1]);
  }
  return pVector;
}

static void matrixChainOrder(const SmallVector<long, 8> &p,
                             std::vector<std::vector<long>> &m,
                             std::vector<std::vector<long>> &s) {
  size_t n = p.size();
  for (size_t i = 0; i < n; i++)
    m[i][i] = 0;

  size_t j = 0;
  long q = 0;
  for (size_t l = 2; l < n; l++) {
    for (size_t i = 1; i < n - l + 1; i++) {
      j = i + l - 1;
      m[i][j] = std::numeric_limits<long>::max();
      for (size_t k = i; k <= j - 1; k++) {
        q = m[i][k] + m[k + 1][j] + p[i - 1] * p[k] * p[j];
        if (q < m[i][j]) {
          m[i][j] = q;
          s[i][j] = k;
        }
      }
    }
  }
}

static void printOptimalParens(const std::vector<std::vector<long>> &s,
                               size_t i, size_t j) {
  using namespace llvm;
  if (i == j)
    dbgs() << " A_" << i << " ";
  else {
    dbgs() << "(";
    printOptimalParens(s, i, s[i][j]);
    printOptimalParens(s, s[i][j] + 1, j);
    dbgs() << ")";
  }
}

static void getMatmuls(Operation *op, SmallVector<Operation *, 8> &matmuls) {
  if (auto matmulOp = dyn_cast_or_null<MatmulOp>(op)) {
    matmuls.push_back(matmulOp.getOperation());
    auto users = matmulOp.getOutputBuffer(0).getUsers();
    for (const auto user : users) {
      if (op == user)
        continue;
      return getMatmuls(user, matmuls);
    }
  }
}

static void
getDefiningOpForMatmulOutputBuffer(SmallVector<Operation *, 8> &matmuls,
                                   SmallVector<Operation *, 8> &allocs) {
  for (const auto &matmul : matmuls) {
    auto linOp = cast<LinalgOp>(matmul);
    allocs.push_back(linOp.getOutputBuffer(0).getDefiningOp());
  }
}

/// Given two 2-d memref A and B, create a destination memref C
/// which stores the product of A * B.
/// For example, A[i][k] * B[k][j] -> C[i][j].
static mlir::Value createAlloc(Value A, Value B, PatternRewriter &rewriter,
                               Location loc) {
  auto memRefA = A.getType().dyn_cast_or_null<MemRefType>();
  auto memRefB = B.getType().dyn_cast_or_null<MemRefType>();
  assert(memRefA && "expect memref type");
  assert(memRefB && "expect memref type");
  auto shapeA = memRefA.getShape();
  auto shapeB = memRefB.getShape();
  auto elementType = memRefA.getElementType();
  auto outputMemRef =
      MemRefType::get({shapeA[0], shapeB[1]}, elementType, {}, 0);
  return rewriter.create<mlir::AllocOp>(loc, outputMemRef);
}

static void createMatmul(Value alpha, Value beta, Value A, Value B, Value C,
                         PatternRewriter &rewriter, Location loc) {
  rewriter.create<MatmulOp>(loc, beta, alpha, A, B, C);
}

static long getCurrentNumberOfScalarMults(SmallVector<Value, 8> values) {
  assert(values.size() >= 2 && "expect at least two matrices");
  long res = 0;
  size_t size = values.size();
  for (size_t i = 0; i < size; i += 2) {
    auto memRefA = values[i].getType().dyn_cast_or_null<MemRefType>();
    auto memRefB = values[i + 1].getType().dyn_cast_or_null<MemRefType>();
    assert(memRefA && "expect memref type");
    assert(memRefB && "expect memref type");
    auto shapeA = memRefA.getShape();
    auto shapeB = memRefB.getShape();
    res = res + (shapeA[0] * shapeB[0] * shapeB[1]);
  }
  return res;
}

class ChainMatmulOp : public RewritePattern {
private:
  static long currentMin;

public:
  ChainMatmulOp(MLIRContext *context)
      : RewritePattern(MatmulOp::getOperationName(), 0, context) {}

  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    //
    // auto module = op->getParentOfType<mlir::ModuleOp>();
    // module.dump();
    //
    using mlir::matchers::m_Any;
    using mlir::matchers::m_AnyCapture;
    Value A1 = nullptr, A2 = nullptr, A3 = nullptr, A4 = nullptr;
    Value B1 = nullptr, B2 = nullptr;
    Value alpha1 = nullptr, alpha2 = nullptr, alpha3 = nullptr;
    Value beta1 = nullptr, beta2 = nullptr, beta3 = nullptr;
    // clang-format off
    auto matcher = 
      m_Op<linalg::MatmulOp>(m_AnyCapture(alpha1), m_AnyCapture(beta1), m_AnyCapture(A1), m_AnyCapture(A2),
        m_Op<linalg::MatmulOp>(m_AnyCapture(alpha2), m_AnyCapture(beta2), m_AnyCapture(B1), m_AnyCapture(A3),
          m_Op<linalg::MatmulOp>(m_AnyCapture(alpha3), m_AnyCapture(beta3), 
                                 m_AnyCapture(B2), m_AnyCapture(A4), m_Any())));
    // clang-format on
    if (!matcher.match(op))
      return failure();

    // TODO: make sure the output of a previuos matmulOp
    // is not also the output of the current one.

    SmallVector<Value, 8> capturedAllocs = {A1, A2, A3, A4};
    auto pVector = getPVector(capturedAllocs);
    if (!pVector.size())
      return failure();

    const size_t n = pVector.size();
    std::vector<std::vector<long>> m(
        n, std::vector<long>(n, std::numeric_limits<long>::max()));
    std::vector<std::vector<long>> s(
        n, std::vector<long>(n, std::numeric_limits<long>::max()));
    matrixChainOrder(pVector, m, s);
    if (!m.size() || !s.size())
      return failure();

    auto currentNumberScalarsMults =
        getCurrentNumberOfScalarMults({A1, A2, B1, A3, B2, A4});

    LLVM_DEBUG(llvm::dbgs()
               << "Min number scalar multiplications: " << m[1][n - 1] << "\n");
    LLVM_DEBUG(llvm::dbgs() << "Current number scalar multiplications: "
                            << currentNumberScalarsMults << "\n");
    LLVM_DEBUG(printOptimalParens(s, 1, 4));
    LLVM_DEBUG(llvm::dbgs() << "\n\n");
    LLVM_DEBUG(llvm::dbgs() << "Current Min: " << currentMin << "\n");

    if (currentNumberScalarsMults == currentMin) {
      return failure();
    }
    currentMin = m[1][n - 1];

    // TODO: Capture the matmulOp via m_Op
    // matcher.
    SmallVector<Operation *, 8> matmuls = {};
    getMatmuls(op, matmuls);
    if (matmuls.size() != 3) {
      LLVM_DEBUG(llvm::dbgs() << "Expect 3 matmul only");
      return failure();
    }
    // get the operations that create the output
    // buffer for the matmul.
    SmallVector<Operation *, 8> allocs = {};
    getDefiningOpForMatmulOutputBuffer(matmuls, allocs);
    if (!llvm::all_of(allocs, [](Operation *alloc) {
          return dyn_cast_or_null<AllocOp>(alloc);
        })) {
      LLVM_DEBUG(llvm::dbgs() << "Expect only alloc operations as defining op "
                                 "for the matmul output buffer\n");
      return failure();
    }
    auto loc = op->getLoc();
    rewriter.setInsertionPoint(matmuls[matmuls.size() - 1]);

    // re-create alloc operations for the matmuls output buffer
    // considering the new ordering.
    // TODO: get the ordering from the s matrix. For now
    // we go manual.
    auto O1 = createAlloc(A3, A4, rewriter, loc);
    auto O2 = createAlloc(A2, O1, rewriter, loc);
    auto O3 = createAlloc(A1, O2, rewriter, loc);
    createMatmul(alpha1, beta1, A3, A4, O1, rewriter, loc);
    createMatmul(alpha2, beta2, A2, O1, O2, rewriter, loc);
    createMatmul(alpha3, beta3, A1, O2, O3, rewriter, loc);

    // delete all the matmuls and the operations
    // that create their output buffer.
    // TODO: Ensure that the alloc operations
    // do not have any more users before erasing.
    for (const auto matmul : matmuls)
      rewriter.eraseOp(matmul);
    for (const auto alloc : allocs)
      rewriter.eraseOp(alloc);

    return success();
  }
};
long ChainMatmulOp::currentMin = std::numeric_limits<long>::max();

static void LinalgMatmulChainPassImpl(FuncOp op, MLIRContext *context) {
  OwningRewritePatternList patterns;
  patterns.insert<ChainMatmulOp>(context);
  applyPatternsAndFoldGreedily(op, patterns);
}

namespace {

struct LinalgMatmulChainPass
    : public LinalgMatmulChainBase<LinalgMatmulChainPass> {
  void runOnFunction() override {
    LinalgMatmulChainPassImpl(getFunction(), &getContext());
  }
};

} // end namespace.

std::unique_ptr<OperationPass<FuncOp>> mlir::createLinalgMatmulChainPass() {
  return std::make_unique<LinalgMatmulChainPass>();
}
