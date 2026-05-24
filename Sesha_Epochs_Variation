# Sesha's Epochs Variation of the Baseline CNN Model
# Parameter changed: number of training epochs
# This parameter controls how many times the model trains through the full dataset.
# Hypothesis: Increasing the number of epochs may improve accuracy because the model
# has more opportunities to learn image patterns, but training time may also increase.

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import warnings
warnings.filterwarnings("ignore")

from tensorflow.keras.datasets import mnist
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
import numpy as np

# Load MNIST dataset from TensorFlow
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Normalize image pixel values
x_train, x_test = x_train / 255.0, x_test / 255.0

# Reshape images for CNN input
x_train = x_train.reshape(-1, 28, 28, 1)
x_test = x_test.reshape(-1, 28, 28, 1)

# Display the first 10 training images
plt.figure(figsize=(12, 6))

for i in range(10):
    plt.subplot(2, 5, i + 1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)

    # Show image in grayscale
    plt.imshow(x_train[i].reshape(28, 28), cmap="gray")

    # Show actual label
    plt.xlabel(f"Actual: {y_train[i]}")

plt.show()

# Create the CNN model
model = models.Sequential()

# First convolution layer
model.add(
    layers.Conv2D(
        32,
        (3, 3),
        activation='relu',
        input_shape=(28, 28, 1)
    )
)

# First pooling layer
model.add(layers.MaxPooling2D((2, 2)))

# Second convolution layer
model.add(layers.Conv2D(64, (3, 3), activation='relu'))

# Second pooling layer
model.add(layers.MaxPooling2D((2, 2)))

# Third convolution layer
model.add(layers.Conv2D(64, (3, 3), activation='relu'))

# Flatten feature maps
model.add(layers.Flatten())

# Dense hidden layer
model.add(layers.Dense(64, activation='relu'))

# Output layer for digits 0 through 9
model.add(layers.Dense(10, activation='softmax'))

# Display model architecture
model.summary()

# Compile the CNN model
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Train the model on the training data
# Changed epochs from 5 to 8 for this variation
history = model.fit(
    x_train,
    y_train,
    epochs=8,
    validation_data=(x_test, y_test)
)

# Evaluate the model on the test data
test_loss, test_accuracy = model.evaluate(x_test, y_test)

# Print test results
print("Test Loss:", test_loss)
print("Test Accuracy:", test_accuracy)

# Generate predictions using the test data
predictions = model.predict(x_test)

# Display the first 10 test images with predictions
plt.figure(figsize=(12, 6))

for i in range(10):
    plt.subplot(2, 5, i + 1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)

    # Show test image
    plt.imshow(x_test[i].reshape(28, 28), cmap="gray")

    # Get predicted label
    predicted_label = np.argmax(predictions[i])

    # Display predicted and actual labels
    plt.xlabel(
        "Pred: " +
        str(predicted_label) +
        " Actual: " +
        str(y_test[i])
    )

plt.show()
