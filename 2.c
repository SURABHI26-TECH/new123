# =====================================
# Practical 2: Implementing Feedforward Neural Network using Keras & TensorFlow
# =====================================

import tensorflow as tf
from tensorflow import keras
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random

# NOTE: %matplotlib inline works only in Jupyter notebooks
# Remove it if running in other environments
%matplotlib inline  

# ----------------------------
# Load MNIST dataset
# ----------------------------
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

print("Length of training set:", len(x_train))
print("Length of test set:", len(x_test))
print("x_train shape:", x_train.shape)
print("x_test shape:", x_test.shape)

# ----------------------------
# Visualize one example
# ----------------------------
plt.matshow(x_train[0], cmap='gray')
plt.title("Example of MNIST Digit")
plt.show()

# ----------------------------
# Normalize data (0–1 scale)
# ----------------------------
x_train = x_train / 255.0
x_test = x_test / 255.0

# ----------------------------
# Define model
# ----------------------------
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(10, activation='softmax')
])

model.summary()

# ----------------------------
# Compile model
# ----------------------------
model.compile(optimizer='sgd',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# ----------------------------
# Train model
# ----------------------------
history = model.fit(x_train, y_train,
                    validation_data=(x_test, y_test),
                    epochs=10,
                    verbose=2)

# ----------------------------
# Evaluate model
# ----------------------------
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
print("Loss = %.3f" % test_loss)
print("Accuracy = %.3f" % test_acc)

# ----------------------------
# Make a random prediction
# ----------------------------
n = random.randint(0, len(x_test) - 1)
plt.imshow(x_test[n], cmap='gray')
plt.title("Random test image")
plt.show()

predicted_value = model.predict(np.expand_dims(x_test[n], axis=0))
print("Handwritten number in the image is =", np.argmax(predicted_value))

# ----------------------------
# Plot training results
# ----------------------------
plt.figure(figsize=(12, 5))

# Accuracy plot
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train')
plt.plot(history.history['val_accuracy'], label='Validation')
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(loc='upper left')

# Loss plot
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train')
plt.plot(history.history['val_loss'], label='Validation')
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(loc='upper left')

plt.show()

# ----------------------------
# Save and reload model
# ----------------------------
keras_model_path = 'keras_mnist_model.keras'
model.save(keras_model_path)

restored_keras_model = tf.keras.models.load_model(keras_model_path)
print("✅ Model saved and reloaded successfully.")
