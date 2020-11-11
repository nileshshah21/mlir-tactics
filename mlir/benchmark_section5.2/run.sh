#!/usr/bin/env bash

export OMP_NUM_THREADS=1
export LD_LIBRARY_PATH="/usr/local/lib:/opt/intel/mkl/lib/intel64"

targets=(
	"atax.mlir"
	"bicg.mlir"
	"gemver.mlir"
	"gessumv.mlir"
	"mvt.mlir"
	"2mm.mlir" 
	"3mm.mlir" 
	"gemm.mlir" 
	"contraction_ab_acd_dbc.mlir"
	"contraction_ab_cad_dcb.mlir"
	"contraction_abc_acd_db.mlir"
	"contraction_abc_ad_bdc.mlir"
	"contraction_abc_bda_dc.mlir"
	"contraction_abcd_aebf_dfce.mlir"
	"contraction_abcd_aebf_fdec.mlir"
)

echo "Running default Clang"
CFLAGS="-convert-linalg-to-affine-loops -lower-affine -convert-scf-to-std -convert-std-to-llvm"
RUNNER="mlir-cpu-runner -O3 -e main -entry-point-result=void -shared-libs=/llvm-project/build/lib/libmlir_test_cblas_interface.so,/llvm-project/build/lib/libmlir_runner_utils.so"

for bm in "${targets[@]}"; do
  x=`echo $bm | sed -e 's/\.mlir$//g'`
  for i in 1 2 3 4 5; do
    t=`mlir-opt $CFLAGS $bm | $RUNNER`
    x="$x:$t"
  done
  echo "$x"
done &> results_clang.txt

echo "Running Linalg with tile factor of 32"
CFLAGS="-raise-affine-to-linalg -convert-linalg-to-affine-loops -affine-loop-tile="tile-size=32" -lower-affine -convert-linalg-to-llvm -convert-scf-to-std -convert-std-to-llvm"

for bm in "${targets[@]}"; do
  x=`echo $bm | sed -e 's/\.mlir$//g'`
  for i in 1 2 3 4 5; do
    t=`mlir-opt $CFLAGS $bm | $RUNNER`
    x="$x:$t"
  done
  echo "$x"
done &> results_linalg.txt

echo "Running BLAS"
CFLAGS="-test-tactics-blas-cpu -convert-linalg-to-affine-loops -lower-affine -convert-linalg-to-llvm -convert-scf-to-std -convert-std-to-llvm"

for bm in "${targets[@]}"; do
  x=`echo $bm | sed -e 's/\.mlir$//g'`
  for i in 1 2 3 4 5; do
    t=`mlir-opt $CFLAGS $bm | $RUNNER`
    x="$x:$t"
  done
  echo "$x"
done &> results_blas.txt
