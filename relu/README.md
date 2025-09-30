  # ReLU
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
python data.py
g++ -fPIC -shared -o libnpu_relu.so relu_npu.cpp 
python experimental
```

To run ReLU on the NPU without JAX, follow the steps below.
```bash
python helper.py
```
## Notice
When you do make run, an error will appear because the Makefile command does not provide the required input files. This  is happening because in the main flow with JAX, the bridge code relu_npu.cpp receives data from JAX, temporarily stores it in .bin files, and then calls relu.exe to perform the computation. For standalone testing and debugging, the helper script helper.py acts as a launcher: it first generates the necessary data files through a separate script and then executes relu.exe with the correct parameters, allowing the compute kernel to be tested independently of the JAX environment.


Running make run is necessary to ensure that all the required components (the executable and the xclbin) have been compiled correctly and are ready. The error is not a failure of the build process, but rather a confirmation that our program (relu.exe) is working as intended and rejecting incomplete commands, exactly as it was designed to do.
