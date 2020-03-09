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

def testTTGTabcadbdc : Tactics<"C(i, j, k) += A(i, l) * B(j, l, k)", [
  permutationBuilder<Inputs<["B"]>, Outputs<["D"]>, AffineExpression<"{1, 0, 2}">>,
  reshapeBuilder<Inputs<["D"]>, Outputs<["E"]>, AffineExpression<"{1, 2}">>,
  reshapeBuilder<Inputs<["C"]>, Outputs<["F"]>, AffineExpression<"{1, 2}">>,
  matmulBuilder<Inputs<["A", "E"]>, Outputs<["F"]>>,
  reshapeBuilder<Inputs<["F"]>, Outputs<["C"]>, AffineExpression<"output_already_defined">>,
  eraseOpBuilder
]>;

def testliftToMatmulTactic : Tactics<"C(i,j) += A(i,k) * B(k,j)", [
  matmulBuilder<Inputs<["A", "B"]>, Outputs<["C"]>>,
  eraseOpBuilder
]>;

def testTransposeATensorAndLiftToMatmulTactic : Tactics<"C(i,j) += A(k,i) * B(k,j)", [
  permutationBuilder<Inputs<["A"]>, Outputs<["D"]>, AffineExpression<"{1, 0}">>,
  matmulBuilder<Inputs<["D", "B"]>, Outputs<["C"]>>,
  eraseOpBuilder
]>;

/*
TODO: hide eraseOpBuilder.
*/

#endif
