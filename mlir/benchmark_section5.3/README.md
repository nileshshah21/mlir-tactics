
As a case for more progressive raising, we consider the matrix chain
multiplications. Here we want to first raise from Affine to Linalg then at the
Linalg level apply the matrix-chain multiplication optimization.  Our starting
point are chains of matrix multiplications expressed in Affine using ```for```
loops. To raise to Affine, let's type ```mlir-opt -raise-affine-to-linalg
dynprog_loops_4.mlir```. In the output, all the loop-based matmuls have been
replaced by high-level operations ```linalg.matmul```. To apply the chain
reordering let's type: ```mlir-opt -raise-affine-to-linalg dynprog_loops_4.mlir
-linalg-matmul-chain -chain-size=4```. The option ```chain-size``` is used to
select which chain to apply. We support chains of 4,5 or 6 matrices. The other
two chains can be optimized by typing: ```mlir-opt -raise-affine-to-linalg
dynprog_loops_5.mlir -linalg-matmul-chain -chain-size=5``` ```mlir-opt
-raise-affine-to-linalg dynprog_loops_6.mlir -linalg-matmul-chain
-chain-size=6``` To see the optimal parenthesization append
```-debug-only=matmul-chain```

To evaluate the impact of the chain optimizations, let's move to the
```timed``` directory where the optimized chains have been wrapped with timing
utilities. To run type ```run.sh```. We expect Time IP > Time OP.
