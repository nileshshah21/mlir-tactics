//===- mlir_test_cblas_interface.cpp - Simple Blas subset interface -------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Simple Blas subset interface implementation.
//
//===----------------------------------------------------------------------===//

#include "include/mlir_test_cblas_interface.h"
#include "include/mlir_test_cblas.h"
#include <assert.h>
#include <iostream>

#include "dnnl.hpp"
using namespace dnnl;

extern "C" void
_mlir_ciface_linalg_fill_viewf32_f32(StridedMemRefType<float, 0> *X, float f) {
  X->data[X->offset] = f;
}

extern "C" void
_mlir_ciface_linalg_fill_viewsxf32_f32(StridedMemRefType<float, 1> *X,
                                       float f) {
  for (unsigned i = 0; i < X->sizes[0]; ++i)
    *(X->data + X->offset + i * X->strides[0]) = f;
}

extern "C" void
_mlir_ciface_linalg_fill_viewsxsxf32_f32(StridedMemRefType<float, 2> *X,
                                         float f) {
  for (unsigned i = 0; i < X->sizes[0]; ++i)
    for (unsigned j = 0; j < X->sizes[1]; ++j)
      *(X->data + X->offset + i * X->strides[0] + j * X->strides[1]) = f;
}

extern "C" void
_mlir_ciface_linalg_fill_viewsxsxsxf32_f32_f32(StridedMemRefType<float, 3> *X,
                                               float f) {
  for (unsigned i = 0; i < X->sizes[0]; ++i)
    for (unsigned j = 0; j < X->sizes[1]; ++j)
      for (unsigned k = 0; k < X->sizes[2]; ++k)
        *(X->data + X->offset + i * X->strides[0] + j * X->strides[1] +
          k * X->strides[2]) = f;
}

extern "C" void
_mlir_ciface_linalg_copy_viewf32_viewf32(StridedMemRefType<float, 0> *I,
                                         StridedMemRefType<float, 0> *O) {
  O->data[O->offset] = I->data[I->offset];
}

extern "C" void
_mlir_ciface_linalg_copy_viewsxf32_viewsxf32(StridedMemRefType<float, 1> *I,
                                             StridedMemRefType<float, 1> *O) {
  if (I->sizes[0] != O->sizes[0]) {
    std::cerr << "Incompatible strided memrefs\n";
    printMemRefMetaData(std::cerr, *I);
    printMemRefMetaData(std::cerr, *O);
    return;
  }
  for (unsigned i = 0; i < I->sizes[0]; ++i)
    O->data[O->offset + i * O->strides[0]] =
        I->data[I->offset + i * I->strides[0]];
}

extern "C" void _mlir_ciface_linalg_copy_viewsxsxf32_viewsxsxf32(
    StridedMemRefType<float, 2> *I, StridedMemRefType<float, 2> *O) {
  if (I->sizes[0] != O->sizes[0] || I->sizes[1] != O->sizes[1]) {
    std::cerr << "Incompatible strided memrefs\n";
    printMemRefMetaData(std::cerr, *I);
    printMemRefMetaData(std::cerr, *O);
    return;
  }
  auto so0 = O->strides[0], so1 = O->strides[1];
  auto si0 = I->strides[0], si1 = I->strides[1];
  for (unsigned i = 0; i < I->sizes[0]; ++i)
    for (unsigned j = 0; j < I->sizes[1]; ++j)
      O->data[O->offset + i * so0 + j * so1] =
          I->data[I->offset + i * si0 + j * si1];
}

extern "C" void _mlir_ciface_linalg_dot_viewsxf32_viewsxf32_viewf32(
    StridedMemRefType<float, 1> *X, StridedMemRefType<float, 1> *Y,
    StridedMemRefType<float, 0> *Z) {
  if (X->strides[0] != 1 || Y->strides[0] != 1 || X->sizes[0] != Y->sizes[0]) {
    std::cerr << "Incompatible strided memrefs\n";
    printMemRefMetaData(std::cerr, *X);
    printMemRefMetaData(std::cerr, *Y);
    printMemRefMetaData(std::cerr, *Z);
    return;
  }
  Z->data[Z->offset] +=
      mlir_test_cblas_sdot(X->sizes[0], X->data + X->offset, X->strides[0],
                           Y->data + Y->offset, Y->strides[0]);
}

extern "C" void _mlir_ciface_linalg_matmul_viewsxsxf32_viewsxsxf32_viewsxsxf32(
    StridedMemRefType<float, 2> *A, StridedMemRefType<float, 2> *B,
    StridedMemRefType<float, 2> *C) {
  if (A->strides[1] != B->strides[1] || A->strides[1] != C->strides[1] ||
      A->strides[1] != 1 || A->sizes[0] < A->strides[1] ||
      B->sizes[0] < B->strides[1] || C->sizes[0] < C->strides[1] ||
      C->sizes[0] != A->sizes[0] || C->sizes[1] != B->sizes[1] ||
      A->sizes[1] != B->sizes[0]) {
    printMemRefMetaData(std::cerr, *A);
    printMemRefMetaData(std::cerr, *B);
    printMemRefMetaData(std::cerr, *C);
    return;
  }

  mlir_test_cblas_sgemm(
      CBLAS_ORDER::CblasRowMajor, CBLAS_TRANSPOSE::CblasNoTrans,
      CBLAS_TRANSPOSE::CblasNoTrans, C->sizes[0], C->sizes[1], A->sizes[1],
      1.0f, A->data + A->offset, A->strides[0], B->data + B->offset,
      B->strides[0], 1.0f, C->data + C->offset, C->strides[0]);
}

void matmulBlas(StridedMemRefType<float, 2> *C, StridedMemRefType<float, 2> *B,
                StridedMemRefType<float, 2> *A) {
  if (A->strides[1] != B->strides[1] || A->strides[1] != C->strides[1] ||
      A->strides[1] != 1 || A->sizes[0] < A->strides[1] ||
      B->sizes[0] < B->strides[1] || C->sizes[0] < C->strides[1] ||
      C->sizes[0] != A->sizes[0] || C->sizes[1] != B->sizes[1] ||
      A->sizes[1] != B->sizes[0]) {
    printMemRefMetaData(std::cerr, *A);
    printMemRefMetaData(std::cerr, *B);
    printMemRefMetaData(std::cerr, *C);
    return;
  }
  std::cout << "\nA -> \n";
  printMemRefMetaData(std::cerr, *A);
  std::cout << "\nB -> \n";
  printMemRefMetaData(std::cerr, *B);
  std::cout << "\nC -> \n";
  printMemRefMetaData(std::cerr, *C);
  std::cout << "\n";
  size_t M = C->sizes[0];
  size_t N = C->sizes[1];
  size_t K = A->sizes[1];
  size_t lda = K;
  size_t ldb = N;
  size_t ldc = N;

  auto res =
      dnnl_sgemm('N', 'N', M, N, K, 1.0, A->data + A->offset, lda,
                 B->data + B->offset, ldb, 1.0, C->data + C->offset, ldc);
  if (res != dnnl_success)
    assert(0 && "dnnl_sgemm failed");
}

// FIXME use the more conventional A, B and C.
extern "C" void _mlir_ciface_matmul_42x42x42(StridedMemRefType<float, 2> *C,
                                             StridedMemRefType<float, 2> *B,
                                             StridedMemRefType<float, 2> *A) {
  matmulBlas(C, B, A);
}

extern "C" void _mlir_ciface_matmul_2x12x5(StridedMemRefType<float, 2> *C,
                                           StridedMemRefType<float, 2> *B,
                                           StridedMemRefType<float, 2> *A) {
  matmulBlas(C, B, A);
}

extern "C" void
_mlir_ciface_linalg_fill_view42x42xf32_f32(StridedMemRefType<float, 2> *X,
                                           float f) {
  _mlir_ciface_linalg_fill_viewsxsxf32_f32(X, f);
}

extern "C" void
_mlir_ciface_linalg_fill_view3x5x4xf32_f32(StridedMemRefType<float, 3> *X,
                                           float f) {
  _mlir_ciface_linalg_fill_viewsxsxsxf32_f32_f32(X, f);
}

extern "C" void
_mlir_ciface_linalg_fill_view2x3x4xf32_f32(StridedMemRefType<float, 3> *X,
                                           float f) {
  _mlir_ciface_linalg_fill_viewsxsxsxf32_f32_f32(X, f);
}

extern "C" void
_mlir_ciface_linalg_fill_view2x5xf32_f32(StridedMemRefType<float, 2> *X,
                                         float f) {
  _mlir_ciface_linalg_fill_viewsxsxf32_f32(X, f);
}

inline memory::dims shapeToMklDnnDims(const StridedMemRefType<float, 3> *T) {
  memory::dims dims(3);
  for (int d = 0; d < 3; ++d) {
    dims[d] = T->sizes[d];
  }
  return dims;
}

inline memory::dims calculateStrides(const memory::dims &dimsOrder) {
  memory::dims strides(dimsOrder.size());
  int lastDimIdx = dimsOrder.size() - 1;
  strides[lastDimIdx] = 1;
  for (int d = lastDimIdx - 1; d >= 0; d--) {
    strides[d] = strides[d + 1] * dimsOrder[d + 1];
  }
  return strides;
}

inline memory::dims reorderStrides(const memory::dims &strides,
                                   std::vector<int> perm) {
  memory::dims reordered_strides;
  reordered_strides.resize(strides.size());
  for (size_t i = 0; i < strides.size(); ++i) {
    reordered_strides[perm[i]] = strides[i];
  }
  return reordered_strides;
}

void createBlockedMemDescHelper(const memory::dims &dims,
                                const memory::dims &strides,
                                dnnl_memory_desc_t *blocked_md) {
  const int k_num_dims = dims.size();
  dnnl_dim_t input_dims[k_num_dims];
  dnnl_dim_t input_strides[k_num_dims];
  for (int i = 0; i < k_num_dims; ++i) {
    input_dims[i] = dims[i];
    input_strides[i] = strides[i];
  }
  dnnl_memory_desc_init_by_strides(blocked_md, k_num_dims, input_dims, dnnl_f32,
                                   input_strides);
}

inline memory::desc getMemDescr(const memory::dims &dims,
                                const memory::dims &strides) {
  dnnl_memory_desc_t blocked_md;
  createBlockedMemDescHelper(dims, strides, &blocked_md);
  return memory::desc(blocked_md);
}

extern "C" void
_mlir_ciface_transpose_3x5x4_to_5x3x4(StridedMemRefType<float, 3> *S,
                                      StridedMemRefType<float, 3> *D, int *perm,
                                      int s) {
  std::cout << "\nSource -> \n";
  printMemRefMetaData(std::cerr, *S);
  std::cout << "\nDest -> \n";
  printMemRefMetaData(std::cerr, *D);
  std::cout << "\n\n";

  std::vector<int> arrayPerm{};
  for (int i = 0; i < s; i++)
    arrayPerm.push_back(*(perm++));

  std::cout << "\nPermutation -> \n";
  for (const auto elem : arrayPerm)
    std::cout << elem << "\n";

  auto cpu_engine = engine(engine::kind::cpu, 0);
  memory::dims in_dims = shapeToMklDnnDims(S);
  memory::dims out_dims = shapeToMklDnnDims(D);
  memory::dims in_strides = calculateStrides(in_dims);
  memory::dims out_strides =
      reorderStrides(calculateStrides(out_dims), arrayPerm);
  auto inputMemDescr = getMemDescr(in_dims, in_strides);
  auto outputMemDescr = getMemDescr(in_dims, out_strides);
  auto inputMemory = memory(inputMemDescr, cpu_engine, S->data + S->offset);
  auto outputMemory = memory(outputMemDescr, cpu_engine, D->data + D->offset);
  auto r1 = reorder(inputMemory, outputMemory);
  auto stream_cpu = stream(cpu_engine);
  r1.execute(stream_cpu, inputMemory, outputMemory);
}

extern "C" void
_mlir_ciface_reshape_2x3x4_to_2x12(StridedMemRefType<float, 3> *S,
                                   StridedMemRefType<float, 2> *D) {
  memcpy(D->data + D->offset, S->data + S->offset,
         S->sizes[0] * S->sizes[1] * S->sizes[2] * sizeof(float));
}

extern "C" void
_mlir_ciface_reshape_5x3x4_to_5x12(StridedMemRefType<float, 3> *S,
                                   StridedMemRefType<float, 2> *D) {
  memcpy(D->data + D->offset, S->data + S->offset,
         S->sizes[0] * S->sizes[1] * S->sizes[2] * sizeof(float));
}

extern "C" void
_mlir_ciface_reshape_2x12_to_2x3x4(StridedMemRefType<float, 2> *S,
                                   StridedMemRefType<float, 3> *D) {
  memcpy(D->data + D->offset, S->data + S->offset,
         S->sizes[0] * S->sizes[1] * sizeof(float));
}
