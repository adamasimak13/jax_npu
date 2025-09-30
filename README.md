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

## 3. Key Results
**3.1 ReLU**

To validate the architecture, we run the ReLU operation with different data sizes. We compared the performance of a standard CPU with our target NPU in two scenarios. The first one is without JAX and the second one is with JAX. All times are measured in milliseconds (ms). 

<img width="989" height="590" alt="image" src="https://github.com/user-attachments/assets/d37d0ade-6f3e-469d-bad8-7c9437589827" />
Figure 1: ReLU Perfomance on NPU with and without JAX
