#!/usr/bin/env bash

# need to export even if not used here.
export OMP_NUM_THREADS=1
export LD_LIBRARY_PATH="/usr/local/lib:/opt/intel/mkl/lib/intel64"

targets=(
  "dynprog_linalg_timed_4.mlir"
  "dynprog_linalg_timed_opt_4.mlir"
  "dynprog_linalg_timed_5.mlir"
  "dynprog_linalg_timed_opt_5.mlir"
  "dynprog_linalg_timed_6.mlir"
  "dynprog_linalg_timed_opt_6.mlir"
)

echo "Running chains"
CFLAGS="-convert-linalg-to-affine-loops -lower-affine -convert-scf-to-std -convert-std-to-llvm"
RUNNER="mlir-cpu-runner -O3 -e main -entry-point-result=void -shared-libs=/llvm-project/build/lib/libmlir_test_cblas_interface.so,/llvm-project/build/lib/libmlir_runner_utils.so"

for bm in "${targets[@]}"; do
  x=`echo $bm | sed -e 's/\.mlir$//g'`
  for i in 1 2 3 4 5; do
    t=`mlir-opt $CFLAGS $bm | $RUNNER`
    x="$x:$t"
  done
  echo "$x"
done &> results_chain.txt
