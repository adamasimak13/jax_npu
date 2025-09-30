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

<img width="360" height="261" alt="image" src="https://github.com/user-attachments/assets/821733d5-1ed3-459b-9fd0-c462bebe7ce8" />


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
 
