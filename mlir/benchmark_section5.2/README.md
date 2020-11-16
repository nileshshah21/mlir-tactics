In Section 5.1, we demonstrate how we can lift a single matrix multiplication.
Here we generalize our lifting to cover more computational motifs
(matrix-multiplication, matrix-vector products, convolutions and TTGT
conversions of tensor contractions to matrix products). We evaluate two raising
paths: ```MLT-Linalg``` and ```MLT-BLAS```. The former raises to the Linalg
dialect and leverages optimizations already implemented at the Linalg level
(i.e., tiling for caches). The latter emits calls to vendor-optimized routines.

To reproduce figure 9, type: ``` ./run.sh ``` The script will execute first the
```Clang -O3``` baseline, the ```MLT-Linalg``` and finally ```MLT-Blas```. Each
kernel is executed 5 times using single precision-operands and recording the
minimal execution time.

Once finished to inspect the results:
``` 
g++ -std=c++11 print_stats.cpp -o print_stats
./print_stats results_clang.txt
./print_stats results_linalg.txt
./print_stats results_blas.txt
```
We expect: geomean MLT-blas > geomean MLT-Linalg > geomean clang.

To inspect the raisied kernels, type:

```
mlir-opt -raise-affine-to-linalg file.mlir
```

or

```
mlir-opt -raise-affine-to-blas-cpu file.mlir
```

