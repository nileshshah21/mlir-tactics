#!/usr/bin/env bash

rm result*

export OMP_NUM_THREADS=1
export LD_LIBRARY_PATH="/home/20176055/mkl/lib:/home/20176055/mkl-dnn/install/lib"

CFLAGS="-convert-linalg-to-affine-loops -lower-affine -convert-scf-to-std -convert-std-to-llvm"
RUNNER="mlir-cpu-runner -O3 -e main -entry-point-result=void -shared-libs=/home/20176055/llvm-project/build/lib/libmlir_test_cblas_interface.so,/home/20176055/llvm-project/build/lib/libmlir_runner_utils.so"

targets=("TENSORabacddbc.mlir" "TENSORabcacddb.mlir" "TENSORabcadbdc.mlir" "TENSORabcaddcb.mlir" "TENSORabcbdadc.mlir" "TENSORabcdaebfdfce.mlir" "TENSORabcdaebffdec.mlir")

#targets=("TENSORabacddbc.mlir" "TENSORabcacddb.mlir" "TENSORabcadbdc.mlir")

#for bm in "${targets[@]}"; do
#  x=`echo $bm | sed -e 's/\.mlir$//g'`
#  for i in 1 2 3 4 5; do
#    t=`mlir-opt $CFLAGS $bm | $RUNNER`
#    x="$x:$t"
#  done
#  echo "$x"
#done > result_linalg_no_tactics.txt

#CFLAGS="-test-tactics-linalg -convert-linalg-to-affine-loops -lower-affine -convert-scf-to-std -convert-std-to-llvm"

CFLAGS="-test-tactics-linalg -convert-linalg-to-affine-loops -affine-loop-tile="tile-size=32" -lower-affine -convert-linalg-to-llvm -convert-scf-to-std -convert-std-to-llvm"

#CFLAGS="-test-tactics-linalg -convert-linalg-to-affine-loops -lower-affine -convert-scf-to-std -convert-std-to-llvm"

for bm in "${targets[@]}"; do
  echo $bm
  x=`echo $bm | sed -e 's/\.mlir$//g'`
  for i in 1 2 3 4 5; do
    t=`mlir-opt $CFLAGS $bm | $RUNNER`
    x="$x:$t"
  done
  echo "$x"
done #> result_linalg_tactics.txt

#cat result_linalg_tactics.txt
#</dev/tty vimdiff "result_linalg_no_tactics.txt" "result_linalg_tactics.txt"
