# NeuralNetwork-Hardware-Acceleration

## Introduction
This project focuses on developing, optimizing, and accelerating a convolutional neural network (CNN) for the CIFAR-10 dataset. The aim is to achieve high accuracy during training and significantly improve inference throughput using FPGA acceleration.

### Development platform


### Step 1: Train a CNN for CIFAR-10 Classification
#### Objective
Design and train a CNN capable of classifying the CIFAR-10 dataset.

#### Approach
1. Build a CNN using tensorflow.
2. 

### Step2: Accelerate inference of the neural network on the FPGA
#### Objective
Use the FPGA fabric (DPU) of the KRIA KV260 board to accelerate the inference of the trained CNN, maximizing throughput without significant accuracy loss. This step was performed using the Vitis-AI framework.

#### Approach
1. Quantize the model using the Vitis AI Model Quantizer API (Refer to vitis-quantizer.py).
2. Refer to KRIA_SoM_Inference.py to perform inference on the DPU.

Optimization involves a co-design approach where the neural network's design has to be modified based on results obtained in step 2.
The parameters of the model (Weights, biases, activations etc.) are quantized from 32-bit to 8-bit values using the Vitis AI Model Quantizer. Additionally, the Vitis AI Model Optimizer can be used to prune the model (GPU required).

### Observations
1. The throughput mainly depends on the number of parameters.
2. Where does quantization help
3. How to reduce model parameters?

