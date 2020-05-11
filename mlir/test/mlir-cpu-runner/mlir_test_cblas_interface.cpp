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
#include <chrono>
#include <iostream>
#include <string.h>
#include <vector>

#ifdef HAS_CPU_SUPPORT
#include "dnnl.hpp"
using namespace dnnl;
#endif

#ifdef HAS_GPU_SUPPORT
#include <cublas_v2.h>
#include <cuda_runtime.h>
#endif

Timer *Timer::timer_instance = nullptr;

extern "C" void start_timer() { Timer::get_instance()->start_timer(); }

extern "C" void stop_timer() { Timer::get_instance()->stop_timer(); }

extern "C" void
_mlir_ciface_linalg_fill_viewf32_f32(StridedMemRefType<float, 0> *X, float f) {
  X->data[X->offset] = f;
}

extern "C" void
_mlir_ciface_linalg_fill_viewsxf32_f32(StridedMemRefType<float, 1> *X,
                                       float f) {
  for (unsigned i = 0; i < X->sizes[0]; ++i) {
    *(X->data + X->offset + i * X->strides[0]) = f;
    f++;
  }
}

extern "C" void
_mlir_ciface_linalg_fill_viewsxsxf32_f32(StridedMemRefType<float, 2> *X,
                                         float f) {
  for (unsigned i = 0; i < X->sizes[0]; ++i)
    for (unsigned j = 0; j < X->sizes[1]; ++j) {
      *(X->data + X->offset + i * X->strides[0] + j * X->strides[1]) = f;
      f++;
    }
}

extern "C" void
_mlir_ciface_linalg_fill_viewsxsxsxf32_f32_f32(StridedMemRefType<float, 3> *X,
                                               float f) {
  for (unsigned i = 0; i < X->sizes[0]; ++i)
    for (unsigned j = 0; j < X->sizes[1]; ++j)
      for (unsigned k = 0; k < X->sizes[2]; ++k) {
        *(X->data + X->offset + i * X->strides[0] + j * X->strides[1] +
          k * X->strides[2]) = f;
        f++;
      }
}

extern "C" void _mlir_ciface_linalg_fill_viewsxsxsxsxf32_f32_f32_f32(
    StridedMemRefType<float, 4> *X, float f) {
  for (unsigned i = 0; i < X->sizes[0]; ++i)
    for (unsigned j = 0; j < X->sizes[1]; ++j)
      for (unsigned k = 0; k < X->sizes[2]; ++k)
        for (unsigned l = 0; l < X->sizes[3]; l++) {
          *(X->data + X->offset + i * X->strides[0] + j * X->strides[1] +
            k * X->strides[2] + l * X->strides[3]) = f;
          f++;
        }
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

void matmulBlas(int transA, int transB, StridedMemRefType<float, 2> *C,
                StridedMemRefType<float, 2> *A, StridedMemRefType<float, 2> *B,
                int alpha, int beta) {
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
  // std::cout << "\nA -> \n";
  // printMemRefMetaData(std::cerr, *A);
  // std::cout << "\nB -> \n";
  // printMemRefMetaData(std::cerr, *B);
  // std::cout << "\nC -> \n";
  // printMemRefMetaData(std::cerr, *C);
  // std::cout << "\n";
  size_t M = C->sizes[0];
  size_t N = C->sizes[1];
  size_t K = A->sizes[1];
  size_t lda = K;
  size_t ldb = N;
  size_t ldc = N;

  char isTransA = (transA) ? 'T' : 'N';
  char isTransB = (transB) ? 'T' : 'N';
#ifdef HAS_CPU_SUPPORT
  auto res = dnnl_sgemm(isTransA, isTransB, M, N, K, (float)alpha,
                        A->data + A->offset, lda, B->data + B->offset, ldb,
                        (float)beta, C->data + C->offset, ldc);
  if (res != dnnl_success)
    assert(0 && "dnnl_sgemm failed");

  return;
#endif

  assert(0 && "naive gemm not implemented yet");
}

extern "C" void _mlir_ciface_matmul_42x42x42(int transA, int transB,
                                             StridedMemRefType<float, 2> *C,
                                             StridedMemRefType<float, 2> *A,
                                             StridedMemRefType<float, 2> *B,
                                             int64_t alpha, int64_t beta,
                                             int64_t dimForM, int64_t dimForN,
                                             int64_t dimForK) {
  // no need for dimForM, N and K as the memref is 2d.
  matmulBlas(transA, transB, C, A, B, alpha, beta);
}

extern "C" void _mlir_ciface_matmul_800x900x1100(
    int transA, int transB, StridedMemRefType<float, 2> *C,
    StridedMemRefType<float, 2> *A, StridedMemRefType<float, 2> *B,
    int64_t alpha, int64_t beta, int64_t dimForM, int64_t dimForN,
    int64_t dimForK) {
  // no need for dimForM, N and K as the memref is 2d.
  matmulBlas(transA, transB, C, A, B, alpha, beta);
}

extern "C" void _mlir_ciface_matmul_900x1100x1200(
    int transA, int transB, StridedMemRefType<float, 2> *C,
    StridedMemRefType<float, 2> *A, StridedMemRefType<float, 2> *B,
    int64_t alpha, int64_t beta, int64_t dimForM, int64_t dimForN,
    int64_t dimForK) {
  // no need for dimForM, N and K as the memref is 2d.
  matmulBlas(transA, transB, C, A, B, alpha, beta);
}

extern "C" void _mlir_ciface_matmul_800x900x1000(
    int transA, int transB, StridedMemRefType<float, 2> *C,
    StridedMemRefType<float, 2> *A, StridedMemRefType<float, 2> *B,
    int64_t alpha, int64_t beta, int64_t dimForM, int64_t dimForN,
    int64_t dimForK) {
  // no need for dimForM, N and K as the memref is 2d.
  matmulBlas(transA, transB, C, A, B, alpha, beta);
}

extern "C" void _mlir_ciface_matmul_800x1100x900(
    int transA, int transB, StridedMemRefType<float, 2> *C,
    StridedMemRefType<float, 2> *A, StridedMemRefType<float, 2> *B,
    int64_t alpha, int64_t beta, int64_t dimForM, int64_t dimForN,
    int64_t dimForK) {
  // no need for dimForM, N and K as the memref is 2d.
  matmulBlas(transA, transB, C, A, B, alpha, beta);
}

extern "C" void _mlir_ciface_matmul_800x1200x900(
    int transA, int transB, StridedMemRefType<float, 2> *C,
    StridedMemRefType<float, 2> *A, StridedMemRefType<float, 2> *B,
    int64_t alpha, int64_t beta, int64_t dimForM, int64_t dimForN,
    int64_t dimForK) {
  // no need for dimForM, N and K as the memref is 2d.
  matmulBlas(transA, transB, C, A, B, alpha, beta);
}

extern "C" void _mlir_ciface_matvec_2000x2000x2000(
    StridedMemRefType<float, 1> *x, StridedMemRefType<float, 2> *A,
    StridedMemRefType<float, 1> *y, float alpha, float beta, int transA) {
  // TODO: fill me.
}

extern "C" void _mlir_ciface_matmul_2x12x5(int transA, int transB,
                                           StridedMemRefType<float, 2> *C,
                                           StridedMemRefType<float, 2> *A,
                                           StridedMemRefType<float, 2> *B,
                                           int64_t alpha, int64_t beta,
                                           int64_t dimForM, int64_t dimForN,
                                           int64_t dimForK) {
  matmulBlas(transA, transB, C, A, B, alpha, beta);
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

extern "C" void
_mlir_ciface_linalg_fill_view1200x1200xf32_f32(StridedMemRefType<float, 2> *X,
                                               float f) {
  _mlir_ciface_linalg_fill_viewsxsxf32_f32(X, f);
}

extern "C" void
_mlir_ciface_linalg_fill_view1200x1000xf32_f32(StridedMemRefType<float, 2> *X,
                                               float f) {
  _mlir_ciface_linalg_fill_viewsxsxf32_f32(X, f);
}

#ifdef HAS_CPU_SUPPORT
template <int D>
inline memory::dims shapeToMklDnnDims(const StridedMemRefType<float, D> *T) {
  memory::dims dims(D);
  for (int d = 0; d < D; ++d) {
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
#endif

template <int T, int Z>
void transposeBlas(StridedMemRefType<float, T> *S,
                   StridedMemRefType<float, Z> *D, int *perm, int s) {
  // std::cout << "\nSource -> \n";
  // printMemRefMetaData(std::cerr, *S);
  // std::cout << "\nDest -> \n";
  // printMemRefMetaData(std::cerr, *D);
  // std::cout << "\n\n";

  std::vector<int> arrayPerm{};
  for (int i = 0; i < s; i++)
    arrayPerm.push_back(*(perm++));

    // std::cout << "\nPermutation -> \n";
    // for (const auto elem : arrayPerm)
    //  std::cout << elem << "\n";

#ifdef HAS_CPU_SUPPORT
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
  return;
#endif

  assert(0 && "naive transpose not implemented yet");
}

extern "C" void
_mlir_ciface_transpose_3x5x4_to_5x3x4(StridedMemRefType<float, 3> *S,
                                      StridedMemRefType<float, 3> *D, int *perm,
                                      int s) {
  transposeBlas(S, D, perm, s);
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
_mlir_ciface_reshape_32x32x32x32_to_1024x1024(StridedMemRefType<float, 4> *S,
                                              StridedMemRefType<float, 2> *D) {
  memcpy(D->data + D->offset, S->data + S->offset,
         S->sizes[0] * S->sizes[1] * S->sizes[2] * S->sizes[3] * sizeof(float));
}

extern "C" void
_mlir_ciface_reshape_2x12_to_2x3x4(StridedMemRefType<float, 2> *S,
                                   StridedMemRefType<float, 3> *D) {
  memcpy(D->data + D->offset, S->data + S->offset,
         S->sizes[0] * S->sizes[1] * sizeof(float));
}

extern "C" void
_mlir_ciface_linalg_fill_view2x3xf32_f32(StridedMemRefType<float, 2> *X,
                                         float f) {
  _mlir_ciface_linalg_fill_viewsxsxf32_f32(X, f);
}

extern "C" void
_mlir_ciface_linalg_fill_view2x4x5xf32_f32(StridedMemRefType<float, 3> *X,
                                           float f) {
  _mlir_ciface_linalg_fill_viewsxsxsxf32_f32_f32(X, f);
}

extern "C" void
_mlir_ciface_linalg_fill_view5x3x4xf32_f32(StridedMemRefType<float, 3> *X,
                                           float f) {
  _mlir_ciface_linalg_fill_viewsxsxsxf32_f32_f32(X, f);
}

extern "C" void _mlir_ciface_matmul_2x3x20(int transA, int transB,
                                           StridedMemRefType<float, 2> *C,
                                           StridedMemRefType<float, 2> *A,
                                           StridedMemRefType<float, 2> *B,
                                           int64_t alpha, int64_t beta,
                                           int64_t dimForM, int64_t dimForN,
                                           int64_t dimForK) {
  matmulBlas(transA, transB, C, A, B, alpha, beta);
}

extern "C" void
_mlir_ciface_transpose_5x3x4_to_4x5x3(StridedMemRefType<float, 3> *S,
                                      StridedMemRefType<float, 3> *D, int *perm,
                                      int s) {
  transposeBlas(S, D, perm, s);
}

extern "C" void
_mlir_ciface_reshape_2x4x5_to_2x20(StridedMemRefType<float, 3> *S,
                                   StridedMemRefType<float, 2> *D) {
  memcpy(D->data + D->offset, S->data + S->offset,
         S->sizes[0] * S->sizes[1] * S->sizes[2] * sizeof(float));
}

extern "C" void
_mlir_ciface_reshape_4x5x3_to_20x3(StridedMemRefType<float, 3> *S,
                                   StridedMemRefType<float, 2> *D) {
  memcpy(D->data + D->offset, S->data + S->offset,
         S->sizes[0] * S->sizes[1] * S->sizes[2] * sizeof(float));
}

extern "C" void
_mlir_ciface_linalg_fill_view1024x1024xf32_f32(StridedMemRefType<float, 2> *X,
                                               float f) {
  _mlir_ciface_linalg_fill_viewsxsxf32_f32(X, f);
}

extern "C" void _mlir_ciface_matmul_1024x1024x1024(
    int transA, int transB, StridedMemRefType<float, 2> *C,
    StridedMemRefType<float, 2> *A, StridedMemRefType<float, 2> *B,
    int64_t alpha, int64_t beta, int64_t dimForM, int64_t dimForN,
    int64_t dimForK) {
  matmulBlas(transA, transB, C, A, B, alpha, beta);
}

extern "C" void
_mlir_ciface_transpose_32x1024x32_to_32x32x1024(StridedMemRefType<float, 3> *S,
                                                StridedMemRefType<float, 3> *D,
                                                int *perm, int s) {
  transposeBlas(S, D, perm, s);
}

extern "C" void
_mlir_ciface_linalg_fill_view1024x32x32xf32_f32(StridedMemRefType<float, 3> *X,
                                                float f) {
  _mlir_ciface_linalg_fill_viewsxsxsxf32_f32_f32(X, f);
}

extern "C" void
_mlir_ciface_linalg_fill_view32x1024x32xf32_f32(StridedMemRefType<float, 3> *X,
                                                float f) {
  _mlir_ciface_linalg_fill_viewsxsxsxf32_f32_f32(X, f);
}

extern "C" void
_mlir_ciface_reshape_32x32x1024_to_1024x1024(StridedMemRefType<float, 3> *S,
                                             StridedMemRefType<float, 2> *D) {
  memcpy(D->data + D->offset, S->data + S->offset,
         S->sizes[0] * S->sizes[1] * S->sizes[2] * sizeof(float));
}

extern "C" void
_mlir_ciface_reshape_1024x32x32_to_1024x1024(StridedMemRefType<float, 3> *S,
                                             StridedMemRefType<float, 2> *D) {
  memcpy(D->data + D->offset, S->data + S->offset,
         S->sizes[0] * S->sizes[1] * S->sizes[2] * sizeof(float));
}

extern "C" void
_mlir_ciface_reshape_1024x1024_to_32x1024x32(StridedMemRefType<float, 2> *S,
                                             StridedMemRefType<float, 3> *D) {
  memcpy(D->data + D->offset, S->data + S->offset,
         S->sizes[0] * S->sizes[1] * sizeof(float));
}

extern "C" void
_mlir_ciface_linalg_fill_view32x32x1024xf32_f32(StridedMemRefType<float, 3> *X,
                                                float f) {
  _mlir_ciface_linalg_fill_viewsxsxsxf32_f32_f32(X, f);
}

extern "C" void
_mlir_ciface_transpose_32x1024x32_to_1024x32x32(StridedMemRefType<float, 3> *S,
                                                StridedMemRefType<float, 3> *D,
                                                int *perm, int s) {
  transposeBlas(S, D, perm, s);
}

extern "C" void
_mlir_ciface_reshape_1024x1024_to_1024x32x32(StridedMemRefType<float, 2> *S,
                                             StridedMemRefType<float, 3> *D) {
  memcpy(D->data + D->offset, S->data + S->offset,
         S->sizes[0] * S->sizes[1] * sizeof(float));
}

extern "C" void
_mlir_ciface_transpose_32x32x1024_to_32x32x1024(StridedMemRefType<float, 3> *S,
                                                StridedMemRefType<float, 3> *D,
                                                int *perm, int s) {
  transposeBlas(S, D, perm, s);
}

extern "C" void
_mlir_ciface_reshape_1024x1024_to_32x32x1024(StridedMemRefType<float, 2> *S,
                                             StridedMemRefType<float, 3> *D) {
  memcpy(D->data + D->offset, S->data + S->offset,
         S->sizes[0] * S->sizes[1] * sizeof(float));
}

extern "C" void
_mlir_ciface_transpose_32x32x1024_to_32x1024x32(StridedMemRefType<float, 3> *S,
                                                StridedMemRefType<float, 3> *D,
                                                int *perm, int s) {
  transposeBlas(S, D, perm, s);
}

extern "C" void _mlir_ciface_transpose_32x32x32x32_to_32x32x32x32(
    StridedMemRefType<float, 4> *S, StridedMemRefType<float, 4> *D, int *perm,
    int s) {
  transposeBlas(S, D, perm, s);
}

extern "C" void
_mlir_ciface_linalg_fill_view32x32x32x32xf32_f32(StridedMemRefType<float, 4> *X,
                                                 float f) {
  _mlir_ciface_linalg_fill_viewsxsxsxsxf32_f32_f32_f32(X, f);
}

extern "C" void
_mlir_ciface_reshape_32x32x32x32_to_1024x32x32(StridedMemRefType<float, 4> *S,
                                               StridedMemRefType<float, 3> *D) {
  memcpy(D->data + D->offset, S->data + S->offset,
         S->sizes[0] * S->sizes[1] * S->sizes[2] * S->sizes[3] * sizeof(float));
}

extern "C" void
_mlir_ciface_reshape_1024x1024_to_32x32x32x32(StridedMemRefType<float, 2> *S,
                                              StridedMemRefType<float, 4> *D) {
  memcpy(D->data + D->offset, S->data + S->offset,
         S->sizes[0] * S->sizes[1] * sizeof(float));
}

extern "C" void
_mlir_ciface_linalg_fill_view32x32xf32_f32(StridedMemRefType<float, 2> *X,
                                           float f) {
  _mlir_ciface_linalg_fill_viewsxsxf32_f32(X, f);
}

extern "C" void
_mlir_ciface_linalg_fill_view32x64xf32_f32(StridedMemRefType<float, 2> *X,
                                           float f) {
  _mlir_ciface_linalg_fill_viewsxsxf32_f32(X, f);
}

extern "C" void
_mlir_ciface_linalg_fill_view64x32xf32_f32(StridedMemRefType<float, 2> *X,
                                           float f) {
  _mlir_ciface_linalg_fill_viewsxsxf32_f32(X, f);
}

extern "C" void
_mlir_ciface_linalg_fill_view900x1200xf32_f32(StridedMemRefType<float, 2> *X,
                                              float f) {
  _mlir_ciface_linalg_fill_viewsxsxf32_f32(X, f);
}

extern "C" void
_mlir_ciface_linalg_fill_view1100x900xf32_f32(StridedMemRefType<float, 2> *X,
                                              float f) {
  _mlir_ciface_linalg_fill_viewsxsxf32_f32(X, f);
}

extern "C" void
_mlir_ciface_linalg_fill_view800x1100xf32_f32(StridedMemRefType<float, 2> *X,
                                              float f) {
  _mlir_ciface_linalg_fill_viewsxsxf32_f32(X, f);
}

extern "C" void
_mlir_ciface_linalg_fill_view800x1200xf32_f32(StridedMemRefType<float, 2> *X,
                                              float f) {
  _mlir_ciface_linalg_fill_viewsxsxf32_f32(X, f);
}

extern "C" void
_mlir_ciface_linalg_fill_view800x900xf32_f32(StridedMemRefType<float, 2> *X,
                                             float f) {
  _mlir_ciface_linalg_fill_viewsxsxf32_f32(X, f);
}

extern "C" void
_mlir_ciface_linalg_fill_view900x1100xf32_f32(StridedMemRefType<float, 2> *X,
                                              float f) {
  _mlir_ciface_linalg_fill_viewsxsxf32_f32(X, f);
}

extern "C" void
_mlir_ciface_linalg_fill_view1000x900xf32_f32(StridedMemRefType<float, 2> *X,
                                              float f) {
  _mlir_ciface_linalg_fill_viewsxsxf32_f32(X, f);
}

extern "C" void
_mlir_ciface_linalg_fill_view1200x1100xf32_f32(StridedMemRefType<float, 2> *X,
                                               float f) {
  _mlir_ciface_linalg_fill_viewsxsxf32_f32(X, f);
}

extern "C" void
_mlir_ciface_linalg_fill_view800x1000xf32_f32(StridedMemRefType<float, 2> *X,
                                              float f) {
  _mlir_ciface_linalg_fill_viewsxsxf32_f32(X, f);
}

extern "C" void
_mlir_ciface_linalg_fill_view2000x2000xf32_f32(StridedMemRefType<float, 2> *X,
                                               float f) {
  _mlir_ciface_linalg_fill_viewsxsxf32_f32(X, f);
}

extern "C" void
_mlir_ciface_linalg_fill_view2000xf32_f32(StridedMemRefType<float, 1> *X,
                                          float f) {
  _mlir_ciface_linalg_fill_viewsxf32_f32(X, f);
}

extern "C" void _mlir_ciface_matmul_32x32x64(int transA, int transB,
                                             StridedMemRefType<float, 2> *C,
                                             StridedMemRefType<float, 2> *A,
                                             StridedMemRefType<float, 2> *B,
                                             int64_t alpha, int64_t beta,
                                             int64_t dimForM, int64_t dimForN,
                                             int64_t dimForK) {
  matmulBlas(transA, transB, C, A, B, alpha, beta);
}

// GPU - Support

#ifdef HAS_GPU_SUPPORT
extern "C" void *_mlir_ciface_allocateMemoryForDevice(int64_t size) {
  std::cout << __func__ << "\n";
  cudaError_t error;
  cudaDeviceProp deviceProp;
  int devId = 0;

  error = cudaGetDeviceProperties(&deviceProp, devId);
  if (error != cudaSuccess) {
    std::cout << "failure!\n";
    assert(0);
  }
  std::cout << deviceProp.name << "\n";
  std::cout << deviceProp.major << "\n";
  std::cout << deviceProp.minor << "\n";

  void *d_A;
  error = cudaMalloc((void **)&d_A, size);
  if (error != cudaSuccess) {
    std::cout << "failure\n";
    assert(0);
  }
  return d_A;
}

extern "C" void
_mlir_ciface_createCallCopyFromHostToDevice(StridedMemRefType<float, 2> *S,
                                            void *D, int64_t size) {
  std::cout << __func__ << "\n";
  cudaError_t error;
  error = cudaMemcpy(D, S->data, size, cudaMemcpyHostToDevice);
  if (error != cudaSuccess) {
    std::cout << "failure\n";
    assert(0);
  }
}

// TODO: use memref to compute lda, ldb, ldc, N, M, K.
extern "C" void
_mlir_ciface_createCallToCublasSgemm(void *C, void *A, void *B,
                                     StridedMemRefType<float, 2> *CMemref,
                                     StridedMemRefType<float, 2> *AMemref,
                                     StridedMemRefType<float, 2> *BMemref) {
  std::cout << __func__ << "\n";
  cublasStatus_t error;
  cublasHandle_t handle;
  error = cublasCreate(&handle);
  if (error != CUBLAS_STATUS_SUCCESS) {
    std::cout << "failure\n";
    assert(0);
  }

  float alpha = 1.0;
  float beta = 1.0;
  for (size_t i = 0; i < 1; i++) {
    error = cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, 1024, 1024, 1024,
                        &alpha, (float *)B, 1024, (float *)A, 1024, &beta,
                        (float *)C, 1024);
    if (error != CUBLAS_STATUS_SUCCESS) {
      std::cout << "failure\n";
      assert(0);
    }
  }

  cublasDestroy(handle);
}

extern "C" void _mlir_ciface_createCallCopyFromDeviceToHost(
    void *S, StridedMemRefType<float, 2> *D, int64_t size) {
  std::cout << __func__ << std::endl;
  cudaError_t error;
  error = cudaMemcpy(D->data, S, size, cudaMemcpyDeviceToHost);
  if (error != cudaSuccess) {
    std::cout << "failure\n";
    assert(0);
  }
}
#endif
