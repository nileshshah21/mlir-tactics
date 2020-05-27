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
