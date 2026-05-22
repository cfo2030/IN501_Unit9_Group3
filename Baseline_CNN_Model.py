from tensorflow.keras.datasets import mnist
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
import numpy as np

# Load MNIST dataset from tensorflow
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# MNIST dataset preprocessing
x_train, x_test = x_train / 255.0, x_test / 255.0 # normalize image pixel values into floating-point numbers
x_train = x_train.reshape(-1, 28, 28, 1) # Reshape normalized training images (28 x 28 pixels)
x_test = x_test.reshape(-1, 28, 28, 1) # Reshape normalized testing images (28 x 28 pixels)

