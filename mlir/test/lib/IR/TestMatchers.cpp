//===- TestMatchers.cpp - Pass to test matchers ---------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Linalg/IR/LinalgOps.h"
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/Access.h"
#include "mlir/IR/Function.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/MatchersBinaryOp.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/LoopUtils.h"

using namespace mlir;

namespace {
/// This is a test pass for verifying matchers.
struct TestMatchers : public PassWrapper<TestMatchers, FunctionPass> {
  void runOnFunction() override;
};
} // end anonymous namespace

// This could be done better but is not worth the variadic template trouble.
template <typename Matcher>
static unsigned countMatches(FuncOp f, Matcher &matcher) {
  unsigned count = 0;
  f.walk([&count, &matcher](Operation *op) {
    if (matcher.match(op))
      ++count;
  });
  return count;
}

using matchers::m_Any;
using matchers::m_Val;
static void test1(FuncOp f) {
  assert(f.getNumArguments() == 3 && "matcher test funcs must have 3 args");

  auto a = m_Val(f.getArgument(0));
  auto b = m_Val(f.getArgument(1));
  auto c = m_Val(f.getArgument(2));

  auto p0 = m_Op<AddFOp>(); // using 0-arity matcher
  llvm::outs() << "Pattern add(*) matched " << countMatches(f, p0)
               << " times\n";

  auto p1 = m_Op<MulFOp>(); // using 0-arity matcher
  llvm::outs() << "Pattern mul(*) matched " << countMatches(f, p1)
               << " times\n";

  auto p2 = m_Op<AddFOp>(m_Op<AddFOp>(), m_Any());
  llvm::outs() << "Pattern add(add(*), *) matched " << countMatches(f, p2)
               << " times\n";

  auto p3 = m_Op<AddFOp>(m_Any(), m_Op<AddFOp>());
  llvm::outs() << "Pattern add(*, add(*)) matched " << countMatches(f, p3)
               << " times\n";

  auto p4 = m_Op<MulFOp>(m_Op<AddFOp>(), m_Any());
  llvm::outs() << "Pattern mul(add(*), *) matched " << countMatches(f, p4)
               << " times\n";

  auto p5 = m_Op<MulFOp>(m_Any(), m_Op<AddFOp>());
  llvm::outs() << "Pattern mul(*, add(*)) matched " << countMatches(f, p5)
               << " times\n";

  auto p6 = m_Op<MulFOp>(m_Op<MulFOp>(), m_Any());
  llvm::outs() << "Pattern mul(mul(*), *) matched " << countMatches(f, p6)
               << " times\n";

  auto p7 = m_Op<MulFOp>(m_Op<MulFOp>(), m_Op<MulFOp>());
  llvm::outs() << "Pattern mul(mul(*), mul(*)) matched " << countMatches(f, p7)
               << " times\n";

  auto mul_of_mulmul = m_Op<MulFOp>(m_Op<MulFOp>(), m_Op<MulFOp>());
  auto p8 = m_Op<MulFOp>(mul_of_mulmul, mul_of_mulmul);
  llvm::outs()
      << "Pattern mul(mul(mul(*), mul(*)), mul(mul(*), mul(*))) matched "
      << countMatches(f, p8) << " times\n";

  // clang-format off
  auto mul_of_muladd = m_Op<MulFOp>(m_Op<MulFOp>(), m_Op<AddFOp>());
  auto mul_of_anyadd = m_Op<MulFOp>(m_Any(), m_Op<AddFOp>());
  auto p9 = m_Op<MulFOp>(m_Op<MulFOp>(
                     mul_of_muladd, m_Op<MulFOp>()),
                   m_Op<MulFOp>(mul_of_anyadd, mul_of_anyadd));
  // clang-format on
  llvm::outs() << "Pattern mul(mul(mul(mul(*), add(*)), mul(*)), mul(mul(*, "
                  "add(*)), mul(*, add(*)))) matched "
               << countMatches(f, p9) << " times\n";

  auto p10 = m_Op<AddFOp>(a, b);
  llvm::outs() << "Pattern add(a, b) matched " << countMatches(f, p10)
               << " times\n";

  auto p11 = m_Op<AddFOp>(a, c);
  llvm::outs() << "Pattern add(a, c) matched " << countMatches(f, p11)
               << " times\n";

  auto p12 = m_Op<AddFOp>(b, a);
  llvm::outs() << "Pattern add(b, a) matched " << countMatches(f, p12)
               << " times\n";

  auto p13 = m_Op<AddFOp>(c, a);
  llvm::outs() << "Pattern add(c, a) matched " << countMatches(f, p13)
               << " times\n";

  auto p14 = m_Op<MulFOp>(a, m_Op<AddFOp>(c, b));
  llvm::outs() << "Pattern mul(a, add(c, b)) matched " << countMatches(f, p14)
               << " times\n";

  auto p15 = m_Op<MulFOp>(a, m_Op<AddFOp>(b, c));
  llvm::outs() << "Pattern mul(a, add(b, c)) matched " << countMatches(f, p15)
               << " times\n";

  auto mul_of_aany = m_Op<MulFOp>(a, m_Any());
  auto p16 = m_Op<MulFOp>(mul_of_aany, m_Op<AddFOp>(a, c));
  llvm::outs() << "Pattern mul(mul(a, *), add(a, c)) matched "
               << countMatches(f, p16) << " times\n";

  auto p17 = m_Op<MulFOp>(mul_of_aany, m_Op<AddFOp>(c, b));
  llvm::outs() << "Pattern mul(mul(a, *), add(c, b)) matched "
               << countMatches(f, p17) << " times\n";
}

void test2(FuncOp f) {
  auto a = m_Val(f.getArgument(0));
  FloatAttr floatAttr;
  auto p = m_Op<MulFOp>(a, m_Op<AddFOp>(a, m_Constant(&floatAttr)));
  auto p1 = m_Op<MulFOp>(a, m_Op<AddFOp>(a, m_Constant()));
  // Last operation that is not the terminator.
  Operation *lastOp = f.getBody().front().back().getPrevNode();
  if (p.match(lastOp))
    llvm::outs()
        << "Pattern add(add(a, constant), a) matched and bound constant to: "
        << floatAttr.getValueAsDouble() << "\n";
  if (p1.match(lastOp))
    llvm::outs() << "Pattern add(add(a, constant), a) matched\n";
}

template <typename T>
void getNestedLoopsImpl(std::vector<SmallVector<T, 4>> &bands, FuncOp f) {
  auto getLoopNest = [&](T forOp) {
    SmallVector<T, 4> band;
    getPerfectlyNestedLoops(band, forOp);
    bands.push_back(band);
  };
  for (auto &block : f)
    for (auto &op : block)
      if (auto forOp = dyn_cast<T>(op))
        getLoopNest(forOp);
}

void getNestedLoops(std::vector<SmallVector<AffineForOp, 4>> &bands, FuncOp f) {
  getNestedLoopsImpl(bands, f);
}

void getNestedLoops(std::vector<SmallVector<scf::ForOp, 4>> &bands, FuncOp f) {
  getNestedLoopsImpl(bands, f);
}

void test3(FuncOp f) {
  if (f.getNumArguments() != 3)
    llvm_unreachable("matcher test func must have 3 args");
  std::vector<SmallVector<AffineForOp, 4>> bands;
  getNestedLoops(bands, f);
  if (bands.size() != 1)
    llvm_unreachable("expect single loop nest");
  auto loops = bands[0];
  if (loops.size() != 3)
    llvm_unreachable("matcher test func must have 3 loops");
  auto i = loops[0].getInductionVar();
  auto j = loops[1].getInductionVar();
  auto k = loops[2].getInductionVar();

  auto ctx = f.getBody().getContext();
  using namespace matchers;
  {
    AccessPatternContext pctx(ctx);
    auto _i = m_Placeholder();
    auto _j = m_Placeholder();
    auto _k = m_Placeholder();
    auto _A = m_ArrayPlaceholder();
    auto _B = m_ArrayPlaceholder();
    auto _C = m_ArrayPlaceholder();
    auto a = m_Op<AffineLoadOp>(_A({_i, _k}));
    auto b = m_Op<AffineLoadOp>(_B({_k, _j}));
    auto c = m_Op<AffineLoadOp>(_C({_i, _j}));
    auto p1 = m_Op<AddFOp>(c, m_Op<MulFOp>(a, b));
    llvm::outs() << "Pattern add(C(i, j), mul(A(i, k), B(k, j))) matched "
                 << countMatches(f, p1) << " times\n";
    auto matchedI = pctx[_i];
    auto matchedJ = pctx[_j];
    auto matchedK = pctx[_k];
    Value matchedA = nullptr;
    Value matchedB = nullptr;
    Value matchedC = nullptr;
    matchedA = pctx[_A];
    matchedB = pctx[_B];
    matchedC = pctx[_C];
    if ((i != matchedI) || (j != matchedJ) || (k != matchedK))
      llvm_unreachable("matching failed");
    if ((!matchedA) || (!matchedB) || (!matchedC))
      llvm_unreachable("matching failed");
  }
}

void test4(FuncOp f) {
  if (f.getNumArguments() != 3)
    llvm_unreachable("matcher test func must have 3 args");
  std::vector<SmallVector<AffineForOp, 4>> bands;
  getNestedLoops(bands, f);
  if (bands.size() != 1)
    llvm_unreachable("expect single loop nest");
  auto loops = bands[0];
  if (loops.size() != 3)
    llvm_unreachable("matcher test func must have 3 loops");

  auto ctx = f.getBody().getContext();
  using namespace matchers;
  {
    AccessPatternContext pctx(ctx);
    auto _i = m_Placeholder();
    auto _j = m_Placeholder();
    auto _k = m_Placeholder();
    auto a = m_Op<AffineLoadOp>(_i, _k);
    auto b = m_Op<AffineLoadOp>(_k, _j);
    auto bTrans = m_Op<AffineLoadOp>(_j, _k);
    auto c = m_Op<AffineLoadOp>(_i, _j);
    auto p1 = m_Op<AddFOp>(c, m_Op<MulFOp>(a, b));
    auto p2 = m_Op<AddFOp>(c, m_Op<MulFOp>(a, bTrans));
    llvm::outs() << "Pattern add(C(i, j), mul(A(i, k), B(k, j))) matched "
                 << countMatches(f, p1) << " times\n";
    llvm::outs() << "Pattern add(C(i, j), mul(A(i, k), B(j, k))) matched "
                 << countMatches(f, p2) << " times\n";
  }
}

void test5(FuncOp f) {
  if (f.getNumArguments() != 3)
    llvm_unreachable("matcher test func must have 3 args");
  std::vector<SmallVector<scf::ForOp, 4>> bands;
  getNestedLoops(bands, f);
  if (bands.size() != 1)
    llvm_unreachable("expect single loop nest");
  auto loops = bands[0];
  if (loops.size() != 3)
    llvm_unreachable("matcher test func must have 3 loops");
  auto i = loops[0].getInductionVar();
  auto j = loops[1].getInductionVar();
  auto k = loops[2].getInductionVar();

  auto ctx = f.getBody().getContext();
  using namespace matchers;
  {
    AccessPatternContext pctx(ctx);
    auto _i = m_Placeholder();
    auto _j = m_Placeholder();
    auto _k = m_Placeholder();
    auto _A = m_ArrayPlaceholder();
    auto _B = m_ArrayPlaceholder();
    auto _C = m_ArrayPlaceholder();
    auto a = m_Op<LoadOp>(_A({_i, _k}));
    auto b = m_Op<LoadOp>(_B({_k, _j}));
    auto c = m_Op<LoadOp>(_C({_i, _j}));
    auto p1 = m_Op<AddFOp>(c, m_Op<MulFOp>(a, b));
    llvm::outs() << "Pattern add(C(i, j), mul(A(i, k), B(k, j))) matched "
                 << countMatches(f, p1) << " times\n";
    auto matchedI = pctx[_i];
    auto matchedJ = pctx[_j];
    auto matchedK = pctx[_k];
    Value matchedA = nullptr;
    Value matchedB = nullptr;
    Value matchedC = nullptr;
    matchedA = pctx[_A];
    matchedB = pctx[_B];
    matchedC = pctx[_C];
    if ((i != matchedI) || (j != matchedJ) || (k != matchedK))
      llvm_unreachable("matching failed");
    if ((!matchedA) || (!matchedB) || (!matchedC))
      llvm_unreachable("matching failed");
  }
}

void test8(FuncOp f) {
  auto ctx = f.getBody().getContext();
  using namespace matchers;
  {
    AccessPatternContext pctx(ctx);
    auto _i = m_Placeholder();
    auto _j = m_Placeholder();
    auto _A = m_ArrayPlaceholder();
    auto exprInc = m_Op<AffineLoadOp>(_A({_i + 11, _j}));
    auto exprOtherInc = m_Op<AffineLoadOp>(_A({_i + 1, _j}));
    auto exprCoeff = m_Op<AffineLoadOp>(_A({2 * _i, _j}));
    auto exprOtherCoeff = m_Op<AffineLoadOp>(_A({6 * _i, _j}));
    auto exprCoeffAndInc = m_Op<AffineLoadOp>(_A({6 * _i + 3, _j}));
    auto exprOtherCoeffAndInc =
        m_Op<AffineLoadOp>(_A({6 * _i + 5, 3 * _j + 4}));

    llvm::outs() << "Pattern loadOp A(i+11, j) matched "
                 << countMatches(f, exprInc) << " times\n";
    llvm::outs() << "Pattern loadOp A(i+1, j) matched "
                 << countMatches(f, exprOtherInc) << " times\n";
    llvm::outs() << "Pattern loadOp A(2*i, j) matched "
                 << countMatches(f, exprCoeff) << " times\n";
    llvm::outs() << "Pattern loadOp A(6*i, j) matched "
                 << countMatches(f, exprOtherCoeff) << " times\n";
    llvm::outs() << "Pattern loadOp A(6*i+3, j) matched "
                 << countMatches(f, exprCoeffAndInc) << " times\n";
    llvm::outs() << "Pattern loadOp A(6*i+5, 3*j+4) matched "
                 << countMatches(f, exprOtherCoeffAndInc) << " times\n";
  }
}

void test9(FuncOp f) {
  std::vector<SmallVector<AffineForOp, 4>> bands;
  getNestedLoops(bands, f);
  assert(bands.size() == 1 && "expect single band");
  auto loops = bands[0];
  assert(loops.size() == 4 && "expect a 4-d loop nest");

  auto out_h = loops[0].getInductionVar();
  auto out_w = loops[1].getInductionVar();
  auto k_h = loops[2].getInductionVar();
  auto k_w = loops[3].getInductionVar();

  auto ctx = f.getBody().getContext();
  using namespace matchers;
  {
    AccessPatternContext pctx(ctx);
    auto _out_h = m_Placeholder();
    auto _out_w = m_Placeholder();
    auto _k_h = m_Placeholder();
    auto _k_w = m_Placeholder();
    auto _A = m_ArrayPlaceholder();

    auto expr = m_Op<AffineLoadOp>(_A({_out_h + _k_h, _out_w + _k_w}));
    llvm::outs() << "Pattern loadOp A(_out_h + _k_h, _out_w + _k_w) matched "
                 << countMatches(f, expr) << " times\n";

    auto matchedOutH = pctx[_out_h];
    auto matchedOutW = pctx[_out_w];
    auto matchedKW = pctx[_k_w];
    auto matchedKH = pctx[_k_h];

    assert(matchedOutH == out_h && "matching failed");
    assert(matchedOutW == out_w && "matching failed");
    assert(matchedKW == k_w && "matching failed");
    assert(matchedKH == k_h && "matching failed");
  }
}

void test10(FuncOp f) {
  std::vector<SmallVector<AffineForOp, 4>> bands;
  getNestedLoops(bands, f);
  auto loops = bands[0];

  auto ch = loops[0].getInductionVar();
  auto out_h = loops[1].getInductionVar();
  auto out_w = loops[2].getInductionVar();
  auto k_h = loops[3].getInductionVar();
  auto k_w = loops[4].getInductionVar();

  auto ctx = f.getBody().getContext();
  using namespace matchers;
  {
    AccessPatternContext pctx(ctx);
    auto _ch = m_Placeholder();
    auto _out_h = m_Placeholder();
    auto _out_w = m_Placeholder();
    auto _k_h = m_Placeholder();
    auto _k_w = m_Placeholder();
    auto _F = m_ArrayPlaceholder();
    auto _I = m_ArrayPlaceholder();
    auto _O = m_ArrayPlaceholder();

    auto exprFilt = m_Op<AffineLoadOp>(_F({_ch, _k_h, _k_w}));
    auto exprImg = m_Op<AffineLoadOp>(_I({_ch, _out_h + _k_h, _out_w + _k_w}));
    auto exprOut = m_Op<AffineLoadOp>(_O({_out_h, _out_w}));
    auto bodyMatcher =
        m_Op<mlir::AddFOp>(exprOut, m_Op<mlir::MulFOp>(exprFilt, exprImg));
    llvm::outs() << "conv matched " << countMatches(f, bodyMatcher)
                 << " times\n";
    auto matchedCh = pctx[_ch];
    auto matchedOutH = pctx[_out_h];
    auto matchedOutW = pctx[_out_w];
    auto matchedKH = pctx[_k_h];
    auto matchedKW = pctx[_k_w];
    assert(matchedCh == ch);
    assert(matchedOutH == out_h);
    assert(matchedOutW == out_w);
    assert(matchedKW == k_w);
    assert(matchedKH == k_h);
  }
}

void test7(FuncOp f) {
  using matchers::m_AnyCapture;

  Value A1 = nullptr;
  Value B1 = nullptr, B2 = nullptr, B3 = nullptr;

  auto p1 = m_Op<linalg::MatmulOp>(
      m_Any(), m_Any(), m_AnyCapture(A1), m_AnyCapture(B1),
      m_Op<linalg::MatmulOp>(m_Any(), m_Any(), m_Any(), m_AnyCapture(B2),
                             m_Op<linalg::MatmulOp>(m_Any(), m_Any(), m_Any(),
                                                    m_AnyCapture(B3),
                                                    m_Any())));

  llvm::outs() << "Pattern linalg.matmul matched " << countMatches(f, p1)
               << " times\n";
}

void test6(FuncOp f) {
  assert(f.getNumArguments() == 4 && "matcher test func must have 2 args");
  using namespace matchers;

  auto a = m_Val(f.getArgument(0));
  auto b = m_Val(f.getArgument(1));
  auto c = m_Val(f.getArgument(2));
  auto d = m_Val(f.getArgument(3));

  auto p1 = m_AddF(a, b);
  llvm::outs() << "Pattern m_AddF matched " << countMatches(f, p1)
               << " times\n";

  auto p2 = m_AddI(c, d);
  llvm::outs() << "Pattern m_AddI matched " << countMatches(f, p2)
               << " times\n";

  auto p3 = m_MulI(m_AddI(c, d), c);
  llvm::outs() << "Pattern m_MulI(m_AddI(*), *) matched " << countMatches(f, p3)
               << " times\n";

  auto p4 = m_MulF(m_AddF(a, b), a);
  llvm::outs() << "Pattern m_MulF(m_AddF(a, b), a) matched "
               << countMatches(f, p4) << " times\n";

  auto p5 = m_MulF(m_AddF(a, b), b);
  llvm::outs() << "Pattern m_MulF(m_AddF(a, b), b) matched "
               << countMatches(f, p5) << " times\n";
}

void test11(FuncOp f) {
  if (f.getNumArguments() != 3)
    llvm_unreachable("matcher test func must have 3 args");
  std::vector<SmallVector<AffineForOp, 4>> bands;
  getNestedLoops(bands, f);
  if (bands.size() != 1)
    llvm_unreachable("expect single loop nest");
  auto loops = bands[0];
  if (loops.size() != 3)
    llvm_unreachable("matcher test func must have 3 loops");
  auto i = loops[0].getInductionVar();
  auto j = loops[1].getInductionVar();
  auto k = loops[2].getInductionVar();

  auto ctx = f.getBody().getContext();
  using namespace matchers;
  {
    AccessPatternContext pctx(ctx);
    auto _i = m_Placeholder();
    auto _j = m_Placeholder();
    auto _k = m_Placeholder();
    auto _A = m_ArrayPlaceholder();
    auto _B = m_ArrayPlaceholder();
    auto _C = m_ArrayPlaceholder();
    auto a = m_Op<AffineLoadOp>(_A({_i, _k}));
    auto b = m_Op<AffineLoadOp>(_B({_k, _j}));
    auto c = m_Op<AffineLoadOp>(_C({_i, _j}));
    auto p1 = m_Op<AddFOp>(c, m_Op<MulFOp>(a, b));
    auto p2 = m_Op<AddFOp>(m_Op<MulFOp>(a, b), c);
    auto p3 = m_Op<AddFOp>(c, m_Op<MulFOp>(b, a));
    auto p4 = m_Op<AddFOp>(m_Op<MulFOp>(b, a), c);
    llvm::outs() << "Pattern add(mul(B(k, j), A(i, k)), C(i, j)) matched "
                 << countMatches(f, p4) << " times\n";
    pctx.reset();
    llvm::outs() << "Pattern add(C(i, j), mul(B(j, j), A(i, k))) matched "
                 << countMatches(f, p3) << " times\n";
    pctx.reset();
    llvm::outs() << "Pattern add(mul(A(i, k), B(k, j)), C(i, j)) matched "
                 << countMatches(f, p2) << " times\n";
    pctx.reset();
    llvm::outs() << "Pattern add(C(i, j), mul(A(i, k), B(k, j))) matched "
                 << countMatches(f, p1) << " times\n";
    auto matchedI = pctx[_i];
    auto matchedJ = pctx[_j];
    auto matchedK = pctx[_k];
    Value matchedA = nullptr;
    Value matchedB = nullptr;
    Value matchedC = nullptr;
    matchedA = pctx[_A];
    matchedB = pctx[_B];
    matchedC = pctx[_C];
    if ((i != matchedI) || (j != matchedJ) || (k != matchedK))
      llvm_unreachable("matching failed");
    if ((!matchedA) || (!matchedB) || (!matchedC))
      llvm_unreachable("matching failed");
  }
}

void TestMatchers::runOnFunction() {
  auto f = getFunction();
  llvm::outs() << f.getName() << "\n";
  if (f.getName() == "test1")
    test1(f);
  if (f.getName() == "test2")
    test2(f);
  if (f.getName() == "matmul")
    test3(f);
  if (f.getName() == "matmulTransB")
    test4(f);
  if (f.getName() == "matmulLoop")
    test5(f);
  if (f.getName() == "binaryMatchers")
    test6(f);
  if (f.getName() == "chainMatmul")
    test7(f);
  if (f.getName() == "matcherExpr")
    test8(f);
  if (f.getName() == "placeholderEpxr")
    test9(f);
  if (f.getName() == "channelConv")
    test10(f);
  if (f.getName() == "multipleFiring")
    test11(f);
}

namespace mlir {
void registerTestMatchers() {
  PassRegistration<TestMatchers>("test-matchers", "Test C++ pattern matchers.");
}
} // namespace mlir
