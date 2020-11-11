# Experiment for section 5.1

## File

- 2mm.mlir (2mm benchmark from Polybench already represented in Affine)
- 3mm.mlir (3mm benchmark from Polybench already represented in Affine)
- mm.mlir (gemm benchmark from Polybench already represented in Affine)

- mm\_linearized (mm benchmark with linearized accesses as in [Darkent](https://github.com/pjreddie/darknet/blob/4a03d405982aa1e1e911eac42b0ffce29cc8c8ef/src/gemm.c#L74)

## Objective

We aim at reproducing Figure 8.

## Step-by-step

In order to detect a contraction of the form C --> C(i,j) + A(i,k) * B(k,j)
we need first to define a tactic in the TDL language (i.e., Listing 8).
To do so, let's move to the ```tactics_dsl/test``` folder:

``` cd /tactics_dsl/test ```

The GEMM tactic is already written and tested in the ```dsltest.cpp``` file.

``` vim dsltest.cpp ```

Specifically, the test that exercises our tactic is called ```shouldLowerToGemm```.
To run the test and inspect its output (TDS entry) let's type:

```
export DUMP_TEST = 1
./dsltest --gtest_filter=DslTest.shouldLowerToGemm
```

Now that we have generated TDS we can plug-in into MLIR. For
convenience this is already done. 

``` 
vim /llvm-project/mlir/test/lib/Tactics/TestTactics.td 
```

At line 173 we can see our GEMM tactics. To see the generated 
matchers and builders type:

``` 
mlir-tblgen --gen-tactics-linalg TestTactics.td 
```

The generated matchers and builders are visibile in class: 

```
struct tactic1 : public mlir::OpRewritePattern<mlir::AffineForOp> { ... }

```

Note in case you get mlir-tblgen command not found type:

```
export PATH=/llvm-project-install/bin:$PATH
```

We have just seen how we can go from TDL to TDS and then automatically
generate matchers and builders from each TDS entry, we are not ready
to reproduce figure 8. Type:

```
cd /llvm-project/mlir/benchmarki\_section5.1
mlir-opt --raise-affine-to-linalg mm.mlir
```

The body of the ```func @contraction.ab.ac.cd(...) ``` has now
been swapped with a single high-level op ```linalg.matmul```

A similar observation can be made for ```2mm.mlir``` and ```3mm.mlir```,
for ```mm_linearized.mlir``` no detection will occur due to the linearized
accesses.
