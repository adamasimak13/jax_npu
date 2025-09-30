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

g++ -fPIC -shared -o libnpu_relu.so matmul_npu.cpp 
python matmul_jax.py
```
