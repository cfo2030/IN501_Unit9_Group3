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

# Displaying the first 10 images
plt.figure(figsize=(12, 6))
for i in range(10):
    plt.subplot(2, 5, i + 1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(x_train[i].reshape(28, 28), cmap="gray") # Shows the images in grayscale
    plt.xlabel(y_train[i]) # Shows the correct label for the image

plt.show()

# Create the baseline CNN model
model = models.Sequential()

# First convolution layer looking for image patterns
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))

# First pooling layer to reduce image size
model.add(layers.MaxPooling2D((2, 2)))

# Second convolution layer looking for more detailed patterns
model.add(layers.Conv2D(64, (3, 3), activation='relu'))

# Second pooling layer
model.add(layers.MaxPooling2D((2, 2)))

# Third convolution layer looking for even more detailed patterns
model.add(layers.Conv2D(64, (3,3), activation='relu'))

# Flatten the 2D feature maps into a 1D list
model.add(layers.Flatten())

# Dense hidden layer to help the model learn how the detected features connect to each digit
model.add(layers.Dense(64, activation='relu'))

# Output layer with 10 possible digit classes: 0 through 9
model.add(layers.Dense(10, activation='softmax'))

# Display the model architecture
model.summary()
