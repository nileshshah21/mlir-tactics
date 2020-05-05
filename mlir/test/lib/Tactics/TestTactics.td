#ifndef TEST_TACTICS
#define TEST_TACTICS

// describe an affine expression.
class AffineExpression<string map> {
  string affineExpr = map;
}

// describe a list of inputs.
class Inputs<list<string> ins> {
  list<string> inputs = ins;
}

// describe a list of outputs.
class Outputs<list<string> outs> {
  list<string> outputs = outs;
}

// describe transpose values.
// `N` non-transpose
// `T` transpose
class Trans<string value> {
  string trans = value;
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

class matmulBuilder<Trans transa, Trans transb, 
                    M mVal, N nVal, K kVal,
                    Constant constantAlpha, Constant constantBeta,
                    Inputs ins, Outputs outs> : Builder { 
  // name.
  let name = "matmul";
  // body.  
  let body = [{
    
    using namespace mlir::edsc;
    using namespace mlir::edsc::ops;
    ScopedContext scop(rewriter, op.getLoc());
    linalg_generic_matmul(getOperandFromParamsMatmul());
    
  }];
  // is A transpose ?
  Trans transA = transa;
  // is B transpose ?
  Trans transB = transb;
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

class matvecBuilder<Trans transa, Inputs ins, Outputs outs, 
                    Constant constantAlpha, Constant constantBeta> : Builder {
  // name.
  let name = "matvec";
  //body.
  let body = [{}];
  // is A transpose ?
  Trans transA = transa;
  // alpha. 
  Constant alpha = constantAlpha;
  // beta.
  Constant beta = constantBeta;
  // input params.
  let inputs = ins;
  // output params.
  let outputs = outs;
}

class transposeBuilder<Inputs ins, Outputs outs, AffineExpression expr> : Builder {
  // name.
  let name = "transpose";
  // body.
  let body = [{
    
    using namespace mlir::edsc;
    using namespace mlir::edsc::ops;
    using namespace mlir::edsc::intrinsics;
        
  }];
  // input params.
  let inputs = ins;
  // output params.
  let outputs = outs;
  // affine expr.  
  AffineExpression affineExpr = expr;
}

class reshapeBuilder<Inputs ins, Outputs outs, AffineExpression expr> : Builder {
  // name.
  let name = "reshape";
  // body 
  let body = [{}];
  // input params.
  let inputs = ins;
  // output params.
  let outputs = outs;
  // affine expr.
  AffineExpression affineExpr = expr;
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

def testTransposeATensorAndLiftToMatmulTactic : Tactics<"C(i,j) += A(k,i) * B(k,j)", [
  matmulBuilder<Trans<"T">, Trans<"N">, M<1>, N<1>, K<1>, Constant<"1">, Constant<"1">, 
                Inputs<["A", "B"]>, Outputs<["C"]>>,
  eraseOpBuilder
]>;

// experiments as described in Gareev et al.
// tensor ab-ac-cb sizes 1024x1024 - 1024x1024 - 1024x1024
def TENSORabaccb : Tactics<"C(a,b) += A(a,c) * B(c,b)", [
  matmulBuilder<Trans<"N">, Trans<"N">, M<1>, N<1>, K<1>, Constant<"1">, Constant<"1">,
                Inputs<["A", "B"]>, Outputs<["C"]>>,
  eraseOpBuilder
]>;

// tensor ab-acd-dbc sizes 1024x1024 - 1024x32×32 - 32x1024×32
def TENSORabacddbc : Tactics<"C(a, b) += A(a, c, d) * B(d, b, c)", [
  transposeBuilder<Inputs<["B"]>, Outputs<["D"]>, AffineExpression<"{2, 0, 1}">>,
  reshapeBuilder<Inputs<["A"]>, Outputs<["E"]>, AffineExpression<"{1, 2}">>,
  reshapeBuilder<Inputs<["D"]>, Outputs<["F"]>, AffineExpression<"{0, 1}">>,
  matmulBuilder<Trans<"N">, Trans<"N">, M<1>, N<1>, K<1>, Constant<"1">, Constant<"1">,
                Inputs<["E", "F"]>, Outputs<["C"]>>,
  eraseOpBuilder
]>;

// tensor abc-acd-db sizes 32×1024×32 - 32×32×1024 - 1024×1024
def TENSORabcacddb : Tactics<"C(a, b, c) += A(a, c, d) * B(d, b)", [
  transposeBuilder<Inputs<["C"]>, Outputs<["D"]>, AffineExpression<"{0, 2, 1}">>,
  reshapeBuilder<Inputs<["D"]>, Outputs<["E"]>, AffineExpression<"{0, 1}">>,
  reshapeBuilder<Inputs<["A"]>, Outputs<["F"]>, AffineExpression<"{0, 1}">>,
  matmulBuilder<Trans<"N">, Trans<"N">, M<1>, N<1>, K<1>, Constant<"1">, Constant<"1">,
                Inputs<["F", "B"]>, Outputs<["E"]>>, 
  reshapeBuilder<Inputs<["E"]>, Outputs<["D"]>, AffineExpression<"">>,
  transposeBuilder<Inputs<["D"]>, Outputs<["C"]>, AffineExpression<"{0, 2, 1}">>,
  eraseOpBuilder
]>;

// tensor abc-ad-bdc sizes 1024×32×32 - 1024×1024- 32×1024×32
def TENSORabcadbdc : Tactics<"C(a, b, c) += A(a, d) * B(b, d, c)", [
  transposeBuilder<Inputs<["B"]>, Outputs<["D"]>, AffineExpression<"{1, 0, 2}">>,
  reshapeBuilder<Inputs<["D"]>, Outputs<["E"]>, AffineExpression<"{1, 2}">>,
  reshapeBuilder<Inputs<["C"]>, Outputs<["F"]>, AffineExpression<"{1, 2}">>,
  matmulBuilder<Trans<"N">, Trans<"N">, M<1>, N<1>, K<1>, Constant<"1">, Constant<"1">,
                Inputs<["A", "E"]>, Outputs<["F"]>>,
  reshapeBuilder<Inputs<["F"]>, Outputs<["C"]>, AffineExpression<"">>,
  eraseOpBuilder
]>;

// tensor ab-cad-dcb 1024×1024 - 32×1024×32 - 32×32×1024
def TENSORabcaddcb : Tactics<"C(a, b) += A(c, a, d) * B(d, c, b)", [
  transposeBuilder<Inputs<["A"]>, Outputs<["D"]>, AffineExpression<"{1, 0, 2}">>,
  transposeBuilder<Inputs<["B"]>, Outputs<["E"]>, AffineExpression<"{1, 0, 2}">>,
  reshapeBuilder<Inputs<["D"]>, Outputs<["F"]>, AffineExpression<"{1, 2}">>,
  reshapeBuilder<Inputs<["E"]>, Outputs<["H"]>, AffineExpression<"{0, 1}">>,
  matmulBuilder<Trans<"N">, Trans<"N">, M<1>, N<1>, K<1>, Constant<"1">, Constant<"1">,
                Inputs<["F", "H"]>, Outputs<["C"]>>,
  eraseOpBuilder
]>;

// tensor abc-bda-dc 32×32×1024 - 32×1024×32 - 1024×1024
def TENSORabcdbadc : Tactics<"C(a, b, c) += A(b, d, a) * B(d, c)", [
  transposeBuilder<Inputs<["A"]>, Outputs<["D"]>, AffineExpression<"{2, 0, 1}">>,
  reshapeBuilder<Inputs<["C"]>, Outputs<["E"]>, AffineExpression<"{0, 1}">>,
  reshapeBuilder<Inputs<["D"]>, Outputs<["F"]>, AffineExpression<"{0, 1}">>,
  matmulBuilder<Trans<"N">, Trans<"N">, M<1>, N<1>, K<1>, Constant<"1">, Constant<"1">,
                Inputs<["F", "B"]>, Outputs<["E"]>>,
  reshapeBuilder<Inputs<["E"]>, Outputs<["C"]>, AffineExpression<"">>,
  eraseOpBuilder
]>;

// tensor abcd-aebf-dfce 32×32×32x32 - 32×32×32x32 - 32×32x32x32
// TODO: for each tensor we need two reshape operations to bring it 
// from 4d to a 2d tensor. We may want to have a more complex reshape
// builder which takes as input pair of dimensions to be reshaped. 
// For example: 
// reshapeBuilder<Inputs<["C"]>, Outputs<["G"]>, AffineExpression<"{0, 1}, {1, 2}">>
def TENSORabcdaebfdfce : Tactics<"C(a, b, c, d) += A(a, e, b, f) * B(d, f, c, e)", [
  transposeBuilder<Inputs<["A"]>, Outputs<["D"]>, AffineExpression<"{0, 2, 1, 3}">>,
  transposeBuilder<Inputs<["B"]>, Outputs<["E"]>, AffineExpression<"{3, 1, 2, 0}">>,
  
  reshapeBuilder<Inputs<["C"]>, Outputs<["F"]>, AffineExpression<"{0, 1}">>,
  reshapeBuilder<Inputs<["F"]>, Outputs<["G"]>, AffineExpression<"{1, 2}">>, // C
  
  reshapeBuilder<Inputs<["D"]>, Outputs<["H"]>, AffineExpression<"{0, 1}">>,
  reshapeBuilder<Inputs<["H"]>, Outputs<["I"]>, AffineExpression<"{1, 2}">>, // A
  
  reshapeBuilder<Inputs<["E"]>, Outputs<["L"]>, AffineExpression<"{0, 1}">>,
  reshapeBuilder<Inputs<["L"]>, Outputs<["M"]>, AffineExpression<"{1, 2}">>, // B

  matmulBuilder<Trans<"N">, Trans<"N">, M<1>, N<1>, K<1>, Constant<"1">, Constant<"1">,
                Inputs<["I", "M"]>, Outputs<["G"]>>,
  reshapeBuilder<Inputs<["G"]>, Outputs<["C"]>, AffineExpression<"">>,
  eraseOpBuilder
]>;

def TENSORabcdaebffdec : Tactics<"C(a, b, c, d) += A(a, e, b, f) * B(f, d, e, c)", [
  transposeBuilder<Inputs<["A"]>, Outputs<["D"]>, AffineExpression<"{0, 2, 1, 3}">>,
  transposeBuilder<Inputs<["B"]>, Outputs<["E"]>, AffineExpression<"{2, 0, 3, 1}">>,
  
  reshapeBuilder<Inputs<["C"]>, Outputs<["F"]>, AffineExpression<"{0, 1}">>,
  reshapeBuilder<Inputs<["F"]>, Outputs<["G"]>, AffineExpression<"{1, 2}">>, // C
  
  reshapeBuilder<Inputs<["D"]>, Outputs<["H"]>, AffineExpression<"{0, 1}">>,
  reshapeBuilder<Inputs<["H"]>, Outputs<["I"]>, AffineExpression<"{1, 2}">>, // A
  
  reshapeBuilder<Inputs<["E"]>, Outputs<["L"]>, AffineExpression<"{0, 1}">>,
  reshapeBuilder<Inputs<["L"]>, Outputs<["M"]>, AffineExpression<"{1, 2}">>, // B

  matmulBuilder<Trans<"N">, Trans<"N">, M<1>, N<1>, K<1>, Constant<"1">, Constant<"1">,
                Inputs<["I", "M"]>, Outputs<["G"]>>,
  reshapeBuilder<Inputs<["G"]>, Outputs<["C"]>, AffineExpression<"">>,
  eraseOpBuilder
]>;

def MatVec : Tactics<"x(i) += A(i, j) * y(j)", [
  matvecBuilder<Trans<"N">, Inputs<["A", "y"]>, Outputs<["x"]>, Constant<"1">, Constant<"1">>,
  eraseOpBuilder
]>;

def MatVecTrans : Tactics<"x(i) += A(j, i) * y(j)", [
  transposeBuilder<Inputs<["A"]>, Outputs<["D"]>, AffineExpression<"{1, 0}">>,
  matvecBuilder<Trans<"N">, Inputs<["D", "y"]>, Outputs<["x"]>, Constant<"1">, Constant<"1">>,
  eraseOpBuilder
]>;

def MatVecWithConstant : Tactics<"x(i) += alpha * A(i,j) * y(j)", [
  matvecBuilder<Trans<"N">, Inputs<["A", "y"]>, Outputs<["x"]>, Constant<"alpha">, Constant<"1">>,
  eraseOpBuilder
]>;

/*
TODO: hide eraseOpBuilder.
*/

#endif
