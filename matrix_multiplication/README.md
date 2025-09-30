# Matrix Multiplication

We provide instructions on how to run this system.

To run ReLU through JAX on the NPU, follow the steps below.

## STEP 1 
```bash
git clone <repository_url>
cd <project_directory>
```
## STEP 2
```bash
make clean
make
make run
```


## STEP 3
```bash

g++ -fPIC -shared -o libnpu_matmul.so matmul_npu.cpp 
python matmul_jax.py
```

## Notice
When you want to run matrix multiplication with JAX for dimensions 768x768, you need to set M = 256 in the Makefile, since JAX will perform data partitioning and split the matrix into 3 parts. This is done to properly prepare the kernels so that each one receives 256 elements.
