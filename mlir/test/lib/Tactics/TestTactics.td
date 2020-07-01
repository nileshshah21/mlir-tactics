#ifndef TEST_TACTICS
#define TEST_TACTICS

// describe an affine expression.
class StrExpr<string str> {
  string stringExpr = str;
}

// describe a list of inputs.
class Inputs<list<string> ins> {
  list<string> inputs = ins;
}

// describe a list of outputs.
class Outputs<list<string> outs> {
  list<string> outputs = outs;
}

// describe how many tensor dimensions
// are involved in M, N and K for matmul.
// Why this?
//
// This allows us to compute (in some cases)
// a contraction operation between a second order tensor
// and a third order one as matmul, instead of emitting
// reshape operations. For example,
// C(m, n, p) = A(m,k) * B(k, n, p)
// can be executed as:
// GEMM (’N’,’N’, m, np, k, 1, A, lda<1>, B, ldb<1>, 0, C, ldc<1>);
class Dim<int n = 1> {
  int value = n;
}

class M<int v> : Dim {
  let value = v;
}

class N<int v> : Dim {
  let value = v;
}

class K<int v> : Dim {
  let value = v;
}

// describe a constant.
class Constant<string value> {
  string valueConstant = value;
}

class Builder<string n = "", code b = ""> {
  // name.
  string name = n;
  // c++ body.
  code body = b;
  // input params.
  Inputs inputs = ?;
  // output params.
  Outputs outputs = ?;
}

class convBuilder<Inputs ins, Outputs outs, StrExpr pad, StrExpr str> : Builder {
  // name.
  let name = "conv";
  // body.
  let body = [{}];
  // input params.
  let inputs = ins;
  // output params.
  let outputs = outs;
  // padding.
  StrExpr padding = pad;
  // stride.
  StrExpr stride = str;
}

class matmulBuilder<StrExpr transa, StrExpr transb, 
                    M mVal, N nVal, K kVal,
                    Constant constantAlpha, Constant constantBeta,
                    Inputs ins, Outputs outs> : Builder { 
  // name.
  let name = "matmul";
  // body.  
  let body = [{}];
  // is A transpose ?
  StrExpr transA = transa;
  // is B transpose ?
  StrExpr transB = transb;
  // how many dimensions for M-N-K?
  M m = mVal;
  N n = nVal;
  K k = kVal;
  // alpha
  Constant alpha = constantAlpha;
  // beta
  Constant beta = constantBeta;
  // input params.
  let inputs = ins;
  // output params.
  let outputs = outs;
}

class matvecBuilder<StrExpr transa, Inputs ins, Outputs outs, 
                    Constant constantAlpha, Constant constantBeta> : Builder {
  // name.
  let name = "matvec";
  //body.
  let body = [{}];
  // is A transpose ?
  StrExpr transA = transa;
  // alpha. 
  Constant alpha = constantAlpha;
  // beta.
  Constant beta = constantBeta;
  // input params.
  let inputs = ins;
  // output params.
  let outputs = outs;
}

class transposeBuilder<Inputs ins, Outputs outs, StrExpr expr> : Builder {
  // name.
  let name = "transpose";
  // body.
  let body = [{}];
  // input params.
  let inputs = ins;
  // output params.
  let outputs = outs;
  // affine expr.  
  StrExpr affineExpr = expr;
}

class reshapeBuilder<Inputs ins, Outputs outs, StrExpr expr> : Builder {
  // name.
  let name = "reshape";
  // body 
  let body = [{}];
  // input params.
  let inputs = ins;
  // output params.
  let outputs = outs;
  // affine expr.
  StrExpr affineExpr = expr;
}

def eraseOpBuilder : Builder { 
  // name.
  let name = "erase";
  // body.
  let body = [{
    rewriter.eraseOp(op);
  }];
}
    
class Tactics<string p, list<Builder> b = []> {
  string pattern = p;
  list<Builder> builders = b;
}

def testBackend : Tactics<"C(i,j) += A(i,k) * B(k,j) * D(i)", [
  eraseOpBuilder
]>;

def testStrExprposeATensorAndLiftToMatmulTactic : Tactics<"C(i,j) += A(k,i) * B(k,j)", [
  matmulBuilder<StrExpr<"T">, StrExpr<"N">, M<1>, N<1>, K<1>, Constant<"1">, Constant<"1">, 
                Inputs<["A", "B"]>, Outputs<["C"]>>,
  eraseOpBuilder
]>;

// experiments as described in Gareev et al.
// tensor ab-ac-cb sizes 1024x1024 - 1024x1024 - 1024x1024
def TENSORabaccb : Tactics<"C(a,b) += A(a,c) * B(c,b)", [
  matmulBuilder<StrExpr<"N">, StrExpr<"N">, M<1>, N<1>, K<1>, Constant<"1">, Constant<"1">,
                Inputs<["A", "B"]>, Outputs<["C"]>>,
  eraseOpBuilder
]>;

def TENSORabaccbWithConstant : Tactics<"C(i,j) += alpha * A(i,k) * B(k,j)", [
  matmulBuilder<StrExpr<"N">, StrExpr<"N">, M<1>, N<1>, K<1>, Constant<"alpha">, Constant<"1">,
                Inputs<["A", "B"]>, Outputs<["C"]>>,
  eraseOpBuilder
]>;

// tensor ab-acd-dbc sizes 1024x1024 - 1024x32×32 - 32x1024×32
def TENSORabacddbc : Tactics<"C(a, b) += A(a, c, d) * B(d, b, c)", [
  transposeBuilder<Inputs<["B"]>, Outputs<["D"]>, StrExpr<"{2, 0, 1}">>,
  reshapeBuilder<Inputs<["A"]>, Outputs<["E"]>, StrExpr<"{0, {1, 2}}">>,
  reshapeBuilder<Inputs<["D"]>, Outputs<["F"]>, StrExpr<"{{0, 1}, 2}">>,
  matmulBuilder<StrExpr<"N">, StrExpr<"N">, M<1>, N<1>, K<1>, Constant<"1">, Constant<"1">,
                Inputs<["E", "F"]>, Outputs<["C"]>>,
  eraseOpBuilder
]>;

// tensor abc-acd-db sizes 32×1024×32 - 32×32×1024 - 1024×1024
def TENSORabcacddb : Tactics<"C(a, b, c) += A(a, c, d) * B(d, b)", [
  transposeBuilder<Inputs<["C"]>, Outputs<["D"]>, StrExpr<"{0, 2, 1}">>,
  reshapeBuilder<Inputs<["D"]>, Outputs<["E"]>, StrExpr<"{{0, 1}, 2}">>,
  reshapeBuilder<Inputs<["A"]>, Outputs<["F"]>, StrExpr<"{{0, 1}, 2}">>,
  matmulBuilder<StrExpr<"N">, StrExpr<"N">, M<1>, N<1>, K<1>, Constant<"1">, Constant<"1">,
                Inputs<["F", "B"]>, Outputs<["E"]>>, 
  reshapeBuilder<Inputs<["E"]>, Outputs<["D"]>, StrExpr<"{{0, 1}, 2}">>,
  transposeBuilder<Inputs<["D"]>, Outputs<["C"]>, StrExpr<"{0, 2, 1}">>,
  eraseOpBuilder
]>;

// tensor abc-ad-bdc sizes 1024×32×32 - 1024×1024- 32×1024×32
def TENSORabcadbdc : Tactics<"C(a, b, c) += A(a, d) * B(b, d, c)", [
  transposeBuilder<Inputs<["B"]>, Outputs<["D"]>, StrExpr<"{1, 0, 2}">>,
  reshapeBuilder<Inputs<["D"]>, Outputs<["E"]>, StrExpr<"{0, {1, 2}}">>,
  reshapeBuilder<Inputs<["C"]>, Outputs<["F"]>, StrExpr<"{0, {1, 2}}">>,
  matmulBuilder<StrExpr<"N">, StrExpr<"N">, M<1>, N<1>, K<1>, Constant<"1">, Constant<"1">,
                Inputs<["A", "E"]>, Outputs<["F"]>>,
  reshapeBuilder<Inputs<["F"]>, Outputs<["C"]>, StrExpr<"{0, {1, 2}}">>,
  eraseOpBuilder
]>;

// tensor ab-cad-dcb 1024×1024 - 32×1024×32 - 32×32×1024
def TENSORabcaddcb : Tactics<"C(a, b) += A(c, a, d) * B(d, c, b)", [
  transposeBuilder<Inputs<["A"]>, Outputs<["D"]>, StrExpr<"{1, 0, 2}">>,
  transposeBuilder<Inputs<["B"]>, Outputs<["E"]>, StrExpr<"{1, 0, 2}">>,
  reshapeBuilder<Inputs<["D"]>, Outputs<["F"]>, StrExpr<"{0, {1, 2}}">>,
  reshapeBuilder<Inputs<["E"]>, Outputs<["H"]>, StrExpr<"{{0, 1}, 2}">>,
  matmulBuilder<StrExpr<"N">, StrExpr<"N">, M<1>, N<1>, K<1>, Constant<"1">, Constant<"1">,
                Inputs<["F", "H"]>, Outputs<["C"]>>,
  eraseOpBuilder
]>;

// tensor abc-bda-dc 32×32×1024 - 32×1024×32 - 1024×1024
def TENSORabcdbadc : Tactics<"C(a, b, c) += A(b, d, a) * B(d, c)", [
  transposeBuilder<Inputs<["A"]>, Outputs<["D"]>, StrExpr<"{2, 0, 1}">>,
  reshapeBuilder<Inputs<["C"]>, Outputs<["E"]>, StrExpr<"{{0, 1}, 2}">>,
  reshapeBuilder<Inputs<["D"]>, Outputs<["F"]>, StrExpr<"{{0, 1}, 2}">>,
  matmulBuilder<StrExpr<"N">, StrExpr<"N">, M<1>, N<1>, K<1>, Constant<"1">, Constant<"1">,
                Inputs<["F", "B"]>, Outputs<["E"]>>,
  reshapeBuilder<Inputs<["E"]>, Outputs<["C"]>, StrExpr<"{{0, 1}, 2}">>,
  eraseOpBuilder
]>;

// tensor abcd-aebf-dfce 32×32×32x32 - 32×32×32x32 - 32×32x32x32
def TENSORabcdaebfdfce : Tactics<"C(a, b, c, d) += A(a, e, b, f) * B(d, f, c, e)", [
  transposeBuilder<Inputs<["A"]>, Outputs<["D"]>, StrExpr<"{0, 2, 1, 3}">>,
  transposeBuilder<Inputs<["B"]>, Outputs<["E"]>, StrExpr<"{3, 1, 2, 0}">>,
  reshapeBuilder<Inputs<["C"]>, Outputs<["F"]>, StrExpr<"{{0, 1}, {2, 3}}">>,
  reshapeBuilder<Inputs<["D"]>, Outputs<["H"]>, StrExpr<"{{0, 1}, {2, 3}}">>,
  reshapeBuilder<Inputs<["E"]>, Outputs<["L"]>, StrExpr<"{{0, 1}, {2, 3}}">>,
  matmulBuilder<StrExpr<"N">, StrExpr<"N">, M<1>, N<1>, K<1>, Constant<"1">, Constant<"1">,
                Inputs<["H", "L"]>, Outputs<["F"]>>,
  reshapeBuilder<Inputs<["F"]>, Outputs<["C"]>, StrExpr<"{{0, 1}, {2, 3}}">>,
  eraseOpBuilder
]>;


def TENSORabcdaebffdec : Tactics<"C(a, b, c, d) += A(a, e, b, f) * B(f, d, e, c)", [
  transposeBuilder<Inputs<["A"]>, Outputs<["D"]>, StrExpr<"{0, 2, 1, 3}">>,
  transposeBuilder<Inputs<["B"]>, Outputs<["E"]>, StrExpr<"{2, 0, 3, 1}">>,
  reshapeBuilder<Inputs<["C"]>, Outputs<["F"]>, StrExpr<"{{0, 1}, {2, 3}}">>,
  reshapeBuilder<Inputs<["D"]>, Outputs<["H"]>, StrExpr<"{{0, 1}, {2, 3}}">>,
  reshapeBuilder<Inputs<["E"]>, Outputs<["L"]>, StrExpr<"{{0, 1}, {2, 3}}">>,
  matmulBuilder<StrExpr<"N">, StrExpr<"N">, M<1>, N<1>, K<1>, Constant<"1">, Constant<"1">,
                Inputs<["H", "L"]>, Outputs<["F"]>>,
  reshapeBuilder<Inputs<["F"]>, Outputs<["C"]>, StrExpr<"{{0, 1}, {2, 3}}">>,
  eraseOpBuilder
]>;

def MatVec : Tactics<"x(i) += A(i, j) * y(j)", [
  matvecBuilder<StrExpr<"N">, Inputs<["A", "y"]>, Outputs<["x"]>, Constant<"1">, Constant<"1">>,
  eraseOpBuilder
]>;

def MatVecStrExpr : Tactics<"x(i) += A(j, i) * y(j)", [
  matvecBuilder<StrExpr<"T">, Inputs<["A", "y"]>, Outputs<["x"]>, Constant<"1">, Constant<"1">>,
  eraseOpBuilder
]>;

def MatVecWithConstant : Tactics<"x(i) += alpha * A(i,j) * y(j)", [
  matvecBuilder<StrExpr<"N">, Inputs<["A", "y"]>, Outputs<["x"]>, Constant<"alpha">, Constant<"1">>,
  eraseOpBuilder
]>;

def MatVecWithConstantStrExprpose : Tactics<"x(i) += alpha * A(j,i) * y(j)", [
  matvecBuilder<StrExpr<"T">, Inputs<["A", "y"]>, Outputs<["x"]>, Constant<"alpha">, Constant<"1">>, 
  eraseOpBuilder
]>;

def Conv : Tactics<"out(out_h, out_w) += filt(k_h, k_w) * image(out_h + k_h, out_w + k_w)", [
  convBuilder<Inputs<["filt", "image"]>, Outputs<["out"]>, StrExpr<"{1, 1, 1, 1}">, StrExpr<"{0, 0}">>,
  eraseOpBuilder
]>;

/*
TODO: hide eraseOpBuilder.
*/

#endif
