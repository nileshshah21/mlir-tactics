#ifndef TEST_TACTICS
#define TEST_TACTICS

class AffineExpression<string map> {
  string affineExpr = map;
}

class Inputs<list<string> ins> {
  list<string> inputs = ins;
}

class Outputs<list<string> outs> {
  list<string> outputs = outs;
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

class matmulBuilder<Inputs ins, Outputs outs> : Builder { 
  // name.
  let name = "matmul";
  // body.  
  let body = [{
    
    using namespace mlir::edsc;
    using namespace mlir::edsc::ops;
    ScopedContext scop(rewriter, op.getLoc());
    linalg_matmul(
      makeValueHandles(getOperandFromParamsMatmul()));
    
  }];
  // input params.
  let inputs = ins;
  // output params.
  let outputs = outs;
}

class permutationBuilder<Inputs ins, Outputs outs, AffineExpression expr> : Builder {
  // name.
  let name = "permute";
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

// what the user need to write
def testBackend : Tactics<"C(i,j) += A(i,k) * B(k,j) * D(i)", [
  eraseOpBuilder
]>;

def testTransposeATensorAndLiftToMatmulTactic : Tactics<"C(i,j) += A(k,i) * B(k,j)", [
  permutationBuilder<Inputs<["A"]>, Outputs<["D"]>, AffineExpression<"{1, 0}">>,
  matmulBuilder<Inputs<["D", "B"]>, Outputs<["C"]>>,
  eraseOpBuilder
]>;

// experiments as described in Gareev et al.
// tensor ab-ac-cb sizes 1024x1024 - 1024x1024 - 1024x1024
def TENSORabaccb : Tactics<"C(a,b) += A(a,c) * B(c,b)", [
  matmulBuilder<Inputs<["A", "B"]>, Outputs<["C"]>>,
  eraseOpBuilder
]>;

// tensor ab-acd-dbc sizes 1024x1024 - 1024x32×32 - 32x1024×32
def TENSORabacddbc : Tactics<"C(a, b) += A(a, c, d) * B(d, b, c)", [
  permutationBuilder<Inputs<["B"]>, Outputs<["D"]>, AffineExpression<"{2, 0, 1}">>,
  reshapeBuilder<Inputs<["A"]>, Outputs<["E"]>, AffineExpression<"{1, 2}">>,
  reshapeBuilder<Inputs<["D"]>, Outputs<["F"]>, AffineExpression<"{0, 1}">>,
  matmulBuilder<Inputs<["E", "F"]>, Outputs<["C"]>>,
  eraseOpBuilder
]>;

// tensor abc-acd-db sizes 32×1024×32 - 32×32×1024 - 1024×1024
def TENSORabcacddb : Tactics<"C(a, b, c) += A(a, c, d) * B(d, b)", [
  permutationBuilder<Inputs<["C"]>, Outputs<["D"]>, AffineExpression<"{0, 2, 1}">>,
  reshapeBuilder<Inputs<["D"]>, Outputs<["E"]>, AffineExpression<"{0, 1}">>,
  reshapeBuilder<Inputs<["A"]>, Outputs<["F"]>, AffineExpression<"{0, 1}">>,
  matmulBuilder<Inputs<["F", "B"]>, Outputs<["E"]>>, 
  reshapeBuilder<Inputs<["E"]>, Outputs<["D"]>, AffineExpression<"">>,
  permutationBuilder<Inputs<["D"]>, Outputs<["C"]>, AffineExpression<"{0, 2, 1}">>,
  eraseOpBuilder
]>;

// tensor abc-ad-bdc sizes 1024×32×32 - 1024×1024- 32×1024×32
def TENSORabcadbdc : Tactics<"C(a, b, c) += A(a, d) * B(b, d, c)", [
  permutationBuilder<Inputs<["B"]>, Outputs<["D"]>, AffineExpression<"{1, 0, 2}">>,
  reshapeBuilder<Inputs<["D"]>, Outputs<["E"]>, AffineExpression<"{1, 2}">>,
  reshapeBuilder<Inputs<["C"]>, Outputs<["F"]>, AffineExpression<"{1, 2}">>,
  matmulBuilder<Inputs<["A", "E"]>, Outputs<["F"]>>,
  reshapeBuilder<Inputs<["F"]>, Outputs<["C"]>, AffineExpression<"">>,
  eraseOpBuilder
]>;

// tensor ab-cad-dcb 1024×1024 - 32×1024×32 - 32×32×1024
def TENSORabcaddcb : Tactics<"C(a, b) += A(c, a, d) * B(d, c, b)", [
  permutationBuilder<Inputs<["A"]>, Outputs<["D"]>, AffineExpression<"{1, 0, 2}">>,
  permutationBuilder<Inputs<["B"]>, Outputs<["E"]>, AffineExpression<"{1, 0, 2}">>,
  reshapeBuilder<Inputs<["D"]>, Outputs<["F"]>, AffineExpression<"{1, 2}">>,
  reshapeBuilder<Inputs<["E"]>, Outputs<["H"]>, AffineExpression<"{0, 1}">>,
  matmulBuilder<Inputs<["F", "H"]>, Outputs<["C"]>>,
  eraseOpBuilder
]>;

// tensor abc-bda-dc 32×32×1024 - 32×1024×32 - 1024×1024
def TENSORabcdbadc : Tactics<"C(a, b, c) += A(b, d, a) * B(d, c)", [
  permutationBuilder<Inputs<["A"]>, Outputs<["D"]>, AffineExpression<"{2, 0, 1}">>,
  reshapeBuilder<Inputs<["C"]>, Outputs<["E"]>, AffineExpression<"{0, 1}">>,
  reshapeBuilder<Inputs<["D"]>, Outputs<["F"]>, AffineExpression<"{0, 1}">>,
  matmulBuilder<Inputs<["F", "B"]>, Outputs<["E"]>>,
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
  permutationBuilder<Inputs<["A"]>, Outputs<["D"]>, AffineExpression<"{0, 2, 1, 3}">>,
  permutationBuilder<Inputs<["B"]>, Outputs<["E"]>, AffineExpression<"{3, 1, 2, 0}">>,
  
  reshapeBuilder<Inputs<["C"]>, Outputs<["F"]>, AffineExpression<"{0, 1}">>,
  reshapeBuilder<Inputs<["F"]>, Outputs<["G"]>, AffineExpression<"{1, 2}">>, // C
  
  reshapeBuilder<Inputs<["D"]>, Outputs<["H"]>, AffineExpression<"{0, 1}">>,
  reshapeBuilder<Inputs<["H"]>, Outputs<["I"]>, AffineExpression<"{1, 2}">>, // A
  
  reshapeBuilder<Inputs<["E"]>, Outputs<["L"]>, AffineExpression<"{0, 1}">>,
  reshapeBuilder<Inputs<["L"]>, Outputs<["M"]>, AffineExpression<"{1, 2}">>, // B

  matmulBuilder<Inputs<["I", "M"]>, Outputs<["G"]>>,
  reshapeBuilder<Inputs<["G"]>, Outputs<["C"]>, AffineExpression<"">>,
  eraseOpBuilder
]>;

def TENSORabcdaebffdec : Tactics<"C(a, b, c, d) += A(a, e, b, f) * B(f, d, e, c)", [
  permutationBuilder<Inputs<["A"]>, Outputs<["D"]>, AffineExpression<"{0, 2, 1, 3}">>,
  permutationBuilder<Inputs<["B"]>, Outputs<["E"]>, AffineExpression<"{2, 0, 3, 1}">>,
  
  reshapeBuilder<Inputs<["C"]>, Outputs<["F"]>, AffineExpression<"{0, 1}">>,
  reshapeBuilder<Inputs<["F"]>, Outputs<["G"]>, AffineExpression<"{1, 2}">>, // C
  
  reshapeBuilder<Inputs<["D"]>, Outputs<["H"]>, AffineExpression<"{0, 1}">>,
  reshapeBuilder<Inputs<["H"]>, Outputs<["I"]>, AffineExpression<"{1, 2}">>, // A
  
  reshapeBuilder<Inputs<["E"]>, Outputs<["L"]>, AffineExpression<"{0, 1}">>,
  reshapeBuilder<Inputs<["L"]>, Outputs<["M"]>, AffineExpression<"{1, 2}">>, // B

  matmulBuilder<Inputs<["I", "M"]>, Outputs<["G"]>>,
  reshapeBuilder<Inputs<["G"]>, Outputs<["C"]>, AffineExpression<"">>,
  eraseOpBuilder
]>;
 
/*
TODO: hide eraseOpBuilder.
*/

#endif
