import tensorflow as tf
import numpy as np
import os
import pickle
import cv2
from tensorflow_model_optimization.quantization.keras import vitis_inspect
from tensorflow_model_optimization.quantization.keras import vitis_quantize

XOC = True

# Specify the path to the model created with the Tensorflow framework
if XOC: 
    model_path = '/home/s3474283/EMAI_Project/h5models/'
    q_model_path = '/home/s3474283/EMAI_Project/quantized_models/'
tf_model_name = '3LAYER_84_200K.h5'
tfq_model_name = '3LAYER_84_200K_q.h5'
tf_model_path = model_path + tf_model_name 
tfq_model_path = q_model_path + tfq_model_name

# Load the original model
tf_model = tf.keras.models.load_model(tf_model_path, compile=False)
tf_model.summary()

def load_cifar10_from_directory(directory):
    # Load the training and test batches
    def load_batch(batch_file):
        with open(batch_file, 'rb') as f:
            batch = pickle.load(f, encoding='bytes')
        images = batch[b'data']
        labels = batch[b'labels']
        return images, labels

    # Load training data
    x_train, y_train = [], []
    for i in range(1, 6):
        batch_file = os.path.join(directory, f"data_batch_{i}")
        images, labels = load_batch(batch_file)
        x_train.append(images)
        y_train.append(labels)
    
    # Convert list of arrays into a single NumPy array
    x_train = np.concatenate(x_train, axis=0)  # Concatenate along axis 0 (vertical)
    y_train = np.concatenate(y_train, axis=0)

    # Load test data
    x_test, y_test = load_batch(os.path.join(directory, "test_batch"))
    
    # Reshape the data to (num_samples, 32, 32, 3)
    x_train = x_train.reshape(x_train.shape[0], 3, 32, 32).transpose(0, 2, 3, 1)
    x_test = x_test.reshape(x_test.shape[0], 3, 32, 32).transpose(0, 2, 3, 1)
    y_test = np.array(y_test)
    # Return as NumPy arrays
    return (x_train, y_train), (x_test, y_test)

# Specify the path to the extracted CIFAR-10 dataset
directory = 'cifar-10-python/cifar-10-batches-py/'

(x_train, y_train), (x_test, y_test) = load_cifar10_from_directory(directory)
print ("Original shape:", x_train.shape, y_train.shape, x_test.shape, y_test.shape)

# Normalize the RGB images and reshape (no grayscale conversion needed)
x_train_norm = (x_train / 255.0).astype('float32')
x_test_norm = (x_test / 255.0).astype('float32')


mean=np.mean(x_train_norm)
std=np.std(x_train_norm)
x_test_norm=(x_test_norm-mean)/std
x_train_norm=(x_train_norm-mean)/std

# Print shapes and data types after normalization
print("New shape:", x_train_norm.shape, y_train.shape, x_test_norm.shape, y_test.shape)
print("Type:", x_train_norm.dtype, y_train.dtype, x_test_norm.dtype, y_test.dtype)

# One-hot encode labels
y_train = tf.one_hot(y_train.astype(np.int32), depth=10)
y_test = tf.one_hot(y_test.astype(np.int32), depth=10)

print("New labeling(y):", y_train[0])

# Quantization process
input_shape = (32, 32, 3)  # RGB input shape

target = 'DPUCZDX8G_ISA1_B4096'

inspector = vitis_inspect.VitisInspector(target=target)
inspector.inspect_model(tf_model,
                        input_shape=input_shape,
                        dump_model=True,
                        dump_model_file="inspect_model.h5",
                        dump_results=True,
                        dump_results_file="inspect_results.txt",
                        verbose=1)

quantizer = vitis_quantize.VitisQuantizer(tf_model)

# Quantize the model using a small subset of training data
tfq_model = quantizer.quantize_model(
    calib_dataset=x_train_norm[0:10]  # Calibration dataset
)

# Save the quantized model
tfq_model.save(tfq_model_path)

# Compile the quantized model
learning_rate = 0.0001
momentum = 0
epsilon = 1e-08
batch_size_test = 1

tfq_model = tf.keras.models.load_model(tfq_model_path)
tfq_model.compile(
    optimizer=tf.keras.optimizers.RMSprop(
        learning_rate=learning_rate,
        momentum=momentum,
        epsilon=epsilon
    ),
    loss=tf.keras.losses.CategoricalCrossentropy(reduction=tf.keras.losses.Reduction.SUM_OVER_BATCH_SIZE),
    metrics=['acc']
)

# Evaluate the quantized model on the test set
tfq_model.evaluate(x_test_norm, y_test, batch_size=batch_size_test)

tfq_model.summary()

# Dump the quantized model for inspection
quantizer.dump_model(tfq_model, dataset=x_train_norm[0:1], dump_float=True)
