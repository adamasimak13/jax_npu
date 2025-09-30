# Integration and evaluation of JAX framework on NPUs
We provide instructions on how to run this system.

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
python data.py
g++ -fPIC -shared -o libnpu_relu.so relu_npu.cpp 
python experimental
```
