# Integration and evaluation of JAX framework on NPUs
## Submission for the Open Hardware Competition 2025 Track: Accelerated Computing
Author: Adam Asimakopoulos

Supervisor: Asst. Prof. Christoforos Kachris

Affiliation: University of West Attica


## 1. Project Goal
 The main goal of this project is to run JAX programs on
 Neural Processing Units (NPUs). JAX is a high
performance machine learning framework from Google. It
 is officially supported on CPUs, GPUs, and TPUs.
 However, the fast-growing field of specialized machine
 learning hardware, especially NPUs, offers a great chance
 for power-efficient and highly accelerated computation.

<img width="350" height="308" alt="image" src="https://github.com/user-attachments/assets/4214b5e7-05b4-42af-b979-a712ec539cef" />



This project proves the concept that connects JAX's user
friendly API with the low-level, hardware environment of
 NPUs. By doing this, we open up opportunities for the JAX
 ecosystem to take advantage of NPUs for neural network
 inference and training. 

## 2. Achievements
We designed and implemented a system that transfers JAX computations to an NPU. We successfully performed two basic neural network operations on the NPU using JAX, ReLU and Matrix Multiplication (Matmul). 

ReLU (Rectified Linear Unit) is a basic, element-wise activation function. Running this operation successfully confirms the whole data path, from JAX's Python environment to the NPU hardware and back. 

Matrix Multiplication (Matmul) is a computationally heavy process and its also one of the most basic operation for modern neural networks. Its successful execution shows that the architecture can manage complex tasks, making it a practical choice for real-world machine learning models. 

The final result is a system where a user can write standard JAX code. Through our implementation, the computation runs on the NPU without needing changes to the user's high-level model definition. 

 <img width="3840" height="919" alt="Data_partition" src="https://github.com/user-attachments/assets/782c0100-014d-44c6-9e37-66cb40827e98" />

 ## 3. Prerequisites
 The following tools and dependencies are required to set up and execute the project:

 ### Hardware Configuration

- **Processor / NPU System:** AMD Ryzen 9 7940HS

- **Main Memory**: 32 GB DDR5

- **Operating System**: Ubuntu 22.04 LTS

 ### Software Environment
- **Programming Languages:**
  - **C++23:** For the host application controlling the NPU.
  - **Python 3.10:** For the MLIR hardware generation scripts and the other python scripts.


- **Toolkits and Libraries:**
  - **AMD Vitis Unified Software Platform 2023.2:** Provides the aie-compiler as well as the required device drivers for the target NPU.
  - **Xilinx Runtime (XRT) 2.16:** Supplies the C++ runtime interface used by the host application for communication with the NPU hardware.
  - **CMake:** Utilized as the build system for compiling and managing the C++ host application.


## 4. Running
Each implementation folder contains a README file that explains how to run the corresponding implementation.

## 5. Video
A short video introducing our project is available on YouTube.

https://www.youtube.com/watch?v=ddtP1KaPBgw
