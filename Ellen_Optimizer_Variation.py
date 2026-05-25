# Parameter change: Adam optimizer to stochastic gradient descent (SGD) optimizer
# Hypothesis: Model will train faster as SGD updates model weights with small batches instead of the whole training
# dataset; model performance will degrade because SGD does not use an adaptive learning rate like Adam,
# it uses a fixed learning rate that needs to be adjusted by the user for optimal results

from tensorflow.keras.datasets import mnist
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
import numpy as np
import time

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

# Compile the CNN model
model.compile(
    optimizer='sgd',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

start_time = time.time()

# Train the model on the training data
history = model.fit(
    x_train,
    y_train,
    epochs=5,
    validation_data=(x_test, y_test)
)

end_time = time.time()

print(f"Total training time: {end_time-start_time:.2f} seconds")

# Evaluate the model on the test data
test_loss, test_accuracy = model.evaluate(x_test, y_test)

# Print the test results
print("Test Loss:", test_loss)
print("Test Accuracy:", test_accuracy)

# Generate predictions using the test data
predictions = model.predict(x_test)

# Display the first 10 test images with predicted labels
plt.figure(figsize=(12, 6))

for i in range(10):
    plt.subplot(2, 5, i + 1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)

    # Show the test image in grayscale
    plt.imshow(x_test[i].reshape(28, 28), cmap="gray")

    # Get the predicted label from the model
    predicted_label = np.argmax(predictions[i])

    # Show the predicted label and actual label
    plt.xlabel("Pred: " + str(predicted_label) + " Actual: " + str(y_test[i]))

plt.show()