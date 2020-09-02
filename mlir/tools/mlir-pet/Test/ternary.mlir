// RUN: mlir-pet %S/Inputs/ternary.c | FileCheck %s

CHECK:  func @scop_entry(%arg0: memref<1xf32>, %arg1: memref<1xf32>) {
CHECK:    %0 = alloc() : memref<1xf32>
CHECK:    %cst = constant 1.000000e+00 : f32
CHECK:    %c0 = constant 0 : index
CHECK:    affine.store %cst, %0[%c0] : memref<1xf32>
CHECK:    %1 = alloc() : memref<1xf32>
CHECK:    %c0_0 = constant 0 : index
CHECK:    %2 = affine.load %0[%c0_0] : memref<1xf32>
CHECK:    %cst_1 = constant 0.000000e+00 : f32
CHECK:    %3 = cmpf "ogt", %2, %cst_1 : f32
CHECK:    %c0_2 = constant 0 : index
CHECK:    %4 = affine.load %arg0[%c0_2] : memref<1xf32>
CHECK:    %c0_3 = constant 0 : index
CHECK:    %5 = affine.load %arg1[%c0_3] : memref<1xf32>
CHECK:    %6 = select %3, %4, %5 : f32
CHECK:    %c0_4 = constant 0 : index
CHECK:    affine.store %6, %1[%c0_4] : memref<1xf32>
CHECK:    return
CHECK:  }


