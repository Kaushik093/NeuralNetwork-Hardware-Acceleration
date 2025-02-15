# Neural-Network-Hardware-Acceleration

## Project Overview
This project focuses on developing and accelerating a convolutional neural network (CNN) for the CIFAR-10 dataset. The aim is to achieve high accuracy during training and improve throughput while performing inference on an FPGA.

### Frameworks and Hardware Platforms
- **Training**: TensorFlow or PyTorch.
- **FPGA Acceleration**: VITIS-AI on KRIA KV260.

### Step 1: Training a CNN for CIFAR-10 Classification
- **Objective**: Design and train a CNN capable of classifying the CIFAR-10 dataset with a target test accuracy of at least **90%**.
- **Dataset**: CIFAR-10, consisting of 60,000 32x32 color images across 10 classes.
- **Approach**:
  - Build a CNN model using common deep learning frameworks.
  - Implement techniques such as data augmentation, batch normalization, dropout, and learning rate scheduling to enhance model performance.
  - Train the model on the CIFAR-10 dataset and validate its performance on the test set.

### Step 2: Accelerating Inference with FPGA
- **Objective**: Use the FPGA fabric of the KRIA KV260 board to accelerate the inference of the trained CNN, maximizing throughput without significant accuracy loss.
- **Approach**:
  - Optimize the trained CNN for hardware implementation, considering quantization and pruning techniques.
  - Deploy the optimized model on the FPGA using **VITIS-AI** or an equivalent platform.
  - Evaluate the performance in terms of inference throughput (inferences per second) and accuracy.
  - Iterate between hardware and algorithm co-optimization to balance accuracy and throughput.




