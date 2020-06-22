// RUN: mlir-opt %s -mlir-disable-threading=true -test-matchers -o /dev/null 2>&1 | FileCheck %s

func @test1(%a: f32, %b: f32, %c: f32) {
  %0 = addf %a, %b: f32
  %1 = addf %a, %c: f32
  %2 = addf %c, %b: f32
  %3 = mulf %a, %2: f32
  %4 = mulf %3, %1: f32
  %5 = mulf %4, %4: f32
  %6 = mulf %5, %5: f32
  return
}

// CHECK-LABEL: test1
//       CHECK:   Pattern add(*) matched 3 times
//       CHECK:   Pattern mul(*) matched 4 times
//       CHECK:   Pattern add(add(*), *) matched 0 times
//       CHECK:   Pattern add(*, add(*)) matched 0 times
//       CHECK:   Pattern mul(add(*), *) matched 0 times
//       CHECK:   Pattern mul(*, add(*)) matched 2 times
//       CHECK:   Pattern mul(mul(*), *) matched 3 times
//       CHECK:   Pattern mul(mul(*), mul(*)) matched 2 times
//       CHECK:   Pattern mul(mul(mul(*), mul(*)), mul(mul(*), mul(*))) matched 1 times
//       CHECK:   Pattern mul(mul(mul(mul(*), add(*)), mul(*)), mul(mul(*, add(*)), mul(*, add(*)))) matched 1 times
//       CHECK:   Pattern add(a, b) matched 1 times
//       CHECK:   Pattern add(a, c) matched 1 times
//       CHECK:   Pattern add(b, a) matched 0 times
//       CHECK:   Pattern add(c, a) matched 0 times
//       CHECK:   Pattern mul(a, add(c, b)) matched 1 times
//       CHECK:   Pattern mul(a, add(b, c)) matched 0 times
//       CHECK:   Pattern mul(mul(a, *), add(a, c)) matched 1 times
//       CHECK:   Pattern mul(mul(a, *), add(c, b)) matched 0 times

func @test2(%a: f32) -> f32 {
  %0 = constant 1.0: f32
  %1 = addf %a, %0: f32
  %2 = mulf %a, %1: f32
  return %2: f32
}

// CHECK-LABEL: test2
//       CHECK:   Pattern add(add(a, constant), a) matched and bound constant to: 1.000000e+00
//       CHECK:   Pattern add(add(a, constant), a) matched

func @matmul(%A: memref<42x40xf32>, %B: memref<40x41xf32>, %C: memref<42x41xf32>) {
  affine.for %i = 0 to 42 {
    affine.for %j = 0 to 41 {
      affine.for %k = 0 to 40 {
        %0 = affine.load %A[%i, %k] : memref<42x40xf32>
        %1 = affine.load %B[%k, %j] : memref<40x41xf32>
        %2 = affine.load %C[%i, %j] : memref<42x41xf32>
        %3 = mulf %0, %1 : f32
        %4 = addf %2, %3 : f32
        affine.store %4, %C[%i, %j] : memref<42x41xf32>
      }
    }
  }
  return
}

// CHECK-LABEL: matmul
//       CHECK: Pattern add(C(i, j), mul(A(i, k), B(k, j))) matched 1 times

func @matmulTransB(%A: memref<42x40xf32>, %B: memref<40x41xf32>, %C: memref<42x41xf32>) {
  affine.for %i = 0 to 42 {
    affine.for %j = 0 to 41 {
      affine.for %k = 0 to 40 {
        %0 = affine.load %A[%i, %k] : memref<42x40xf32>
        %1 = affine.load %B[%j, %k] : memref<40x41xf32>
        %2 = affine.load %C[%i, %j] : memref<42x41xf32>
        %3 = mulf %0, %1 : f32
        %4 = addf %2, %3 : f32
        affine.store %4, %C[%i, %j] : memref<42x41xf32>
      }
    }
  }
  return
}

// CHECK-LABEL: matmulTransB
//       CHECK: Pattern add(C(i, j), mul(A(i, k), B(k, j))) matched 0 times
//       CHECK: Pattern add(C(i, j), mul(A(i, k), B(j, k))) matched 1 times

func @matmulLoop(%A: memref<42x42xf32>, %B: memref<42x42xf32>, %C: memref<42x42xf32>) {
  %c42 = constant 42 : index
  %c0 = constant 0 : index
  %c1 = constant 1 : index
  scf.for %i = %c0 to %c42 step %c1 {
    scf.for %j = %c0 to %c42 step %c1 {
      scf.for %k = %c0 to %c42 step %c1 {
        %0 = load %A[%i, %k] : memref<42x42xf32>
        %1 = load %B[%k, %j] : memref<42x42xf32>
        %2 = load %C[%i, %j] : memref<42x42xf32>
        %3 = mulf %0, %1 : f32
        %4 = addf %2, %3 : f32
        store %4, %C[%i, %j] : memref<42x42xf32>
      }
    }
  }
  return
}

// CHECK-LABEL: matmulLoop
//       CHECK:  Pattern add(C(i, j), mul(A(i, k), B(k, j))) matched 1 times

func @binaryMatchers(%a: f32, %b: f32, %c: i32, %d: i32) {
  %0 = addf %a, %b: f32
  %1 = addf %b, %a: f32
  %2 = addi %d, %c: i32
  %3 = addi %c, %d: i32
  %4 = muli %3, %c: i32
  %5 = muli %c, %3: i32
  %6 = mulf %0, %a: f32
  return
}

// CHECK-LABEL: binaryMatchers
//       CHECK: Pattern m_AddF matched 1 times
//       CHECK: Pattern m_AddI matched 2 times
//       CHECK: Pattern m_MulI(m_AddI(*), *) matched 2 times
//       CHECK: Pattern m_MulF(m_AddF(a, b), a) matched 1 times
//       CHECK: Pattern m_MulF(m_AddF(a, b), b) matched 0 times

func @chainMatmul() {
    %arg0 = alloc() : memref<800x1100xf32>
    %arg1 = alloc() : memref<1100x900xf32>
    %arg2 = alloc() : memref<900x1200xf32>
    %arg3 = alloc() : memref<1200x100xf32>
    %arg4 = alloc() : memref<800x900xf32>
    %arg5 = alloc() : memref<800x1200xf32>
    %arg6 = alloc() : memref<800x100xf32>

    %cst = constant 0.000000e+00 : f32
    %cst_0 = constant 1.000000e+00 : f32
    affine.for %arg7 = 0 to 800 {
      affine.for %arg8 = 0 to 1100 {
        affine.store %cst, %arg0[%arg7, %arg8] : memref<800x1100xf32>
      }
    }
    affine.for %arg7 = 0 to 1100 {
      affine.for %arg8 = 0 to 900 {
        affine.store %cst, %arg1[%arg7, %arg8] : memref<1100x900xf32>
      }
    }
    affine.for %arg7 = 0 to 900 {
      affine.for %arg8 = 0 to 1200 {
        affine.store %cst, %arg2[%arg7, %arg8] : memref<900x1200xf32>
      }
    }
    affine.for %arg7 = 0 to 1200 {
      affine.for %arg8 = 0 to 1000 {
        affine.store %cst, %arg3[%arg7, %arg8] : memref<1200x100xf32>
      }
    }
    %0 = alloc() : memref<f32>
    %1 = alloc() : memref<f32>
    linalg.fill(%1, %cst_0) : memref<f32>, f32
    linalg.fill(%0, %cst_0) : memref<f32>, f32
    linalg.matmul(%1, %0, %arg0, %arg1, %arg4) :
      memref<f32>, memref<f32>, memref<800x1100xf32>, memref<1100x900xf32>, memref<800x900xf32>
    %2 = alloc() : memref<f32>
    %3 = alloc() : memref<f32>
    linalg.fill(%3, %cst_0) : memref<f32>, f32
    linalg.fill(%2, %cst_0) : memref<f32>, f32
    linalg.matmul(%3, %2, %arg4, %arg2, %arg5) :
      memref<f32>, memref<f32>, memref<800x900xf32>, memref<900x1200xf32>, memref<800x1200xf32>
    %4 = alloc() : memref<f32>
    %5 = alloc() : memref<f32>
    linalg.fill(%5, %cst_0) : memref<f32>, f32
    linalg.fill(%4, %cst_0) : memref<f32>, f32
    linalg.matmul(%5, %4, %arg5, %arg3, %arg6) :
      memref<f32>, memref<f32>, memref<800x1200xf32>, memref<1200x100xf32>, memref<800x100xf32>
    return
}

// CHECK-LABEL: chainMatmul
//       CHECK: Pattern linalg.matmul matched 1 times

func @matcherExpr(%A: memref<1024x1024xf32>) { 
  affine.for %i = 0 to 256 {
    affine.for %j = 0 to 256 {
      %0 = affine.load %A[%i+11, %j] : memref<1024x1024xf32>
      affine.store %0, %A[%i, %j] : memref<1024x1024xf32>
      %1 = affine.load %A[%i+1, %j] : memref<1024x1024xf32>
      affine.store %1, %A[%i, %j] : memref<1024x1024xf32>
      %2 = affine.load %A[2*%i, %j] : memref<1024x1024xf32>
      affine.store %2, %A[%i, %j] : memref<1024x1024xf32>
      %3 = affine.load %A[6*%i+3, %j] : memref<1024x1024xf32>
      affine.store %3, %A[%i, %j] : memref<1024x1024xf32>
      %4 = affine.load %A[6*%i+5, 3*%j+4] : memref<1024x1024xf32>
      affine.store %4, %A[%i, %j] : memref<1024x1024xf32>
    }
  }
  return
}

// CHECK-LABEL: matcherExpr
//       CHECK: Pattern loadOp A(i+11, j) matched 1 times
//       CHECK: Pattern loadOp A(i+1, j) matched 1 times
//       CHECK: Pattern loadOp A(2*i, j) matched 1 times
//       CHECK: Pattern loadOp A(6*i, j) matched 0 times
//       CHECK: Pattern loadOp A(6*i+3, j) matched 1 times
//       CHECK: Pattern loadOp A(6*i+5, 3*j+4) matched 1 times

func @placeholderEpxr(%out: memref<3x3xf32>, %filt: memref<3x3xf32>, %img: memref<5x5xf32>) {
  affine.for %out_h = 0 to 3 {
    affine.for %out_w = 0 to 3 {
      affine.for %k_h = 0 to 3 {
        affine.for %k_w = 0 to 3 {
          %0 = affine.load %out[%out_h, %out_w] : memref<3x3xf32>
          %1 = affine.load %filt[%k_h, %k_w] : memref<3x3xf32>
          %2 = affine.load %img[%out_h + %k_h, %out_w + %k_w] : memref<5x5xf32>
          %3 = mulf %1, %2 : f32
          %4 = addf %3, %0 : f32
          affine.store %4, %out[%out_h, %out_w] : memref<3x3xf32>
        }
      }
    }
  }
  return
}

// CHECK-LABEL: placeholderEpxr
//       CHECK: Pattern loadOp A(_out_h + _k_h, _out_w + _k_w) matched 1 times
