# ===========================================
# Practical 3: Image Classification using CNN on CIFAR-10
# ===========================================

import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt
import numpy as np
import random

print("✅ TensorFlow version:", tf.__version__)

# ===========================================
# 1️⃣ Load CIFAR-10 Dataset
# ===========================================
(x_train, y_train), (x_test, y_test) = datasets.cifar10.load_data()

print("Training set shape:", x_train.shape)
print("Test set shape:", x_test.shape)

# ===========================================
# 2️⃣ Class Names
# ===========================================
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']

# ===========================================
# 3️⃣ Visualize Sample Images
# ===========================================
plt.figure(figsize=(10, 3))
for i in range(10):
    plt.subplot(2, 5, i+1)
    plt.imshow(x_train[i])
    plt.title(class_names[int(y_train[i])])
    plt.axis('off')
plt.show()

# ===========================================
# 4️⃣ Data Normalization
# ===========================================
x_train = x_train / 255.0
x_test = x_test / 255.0

# Flatten label arrays
y_train = y_train.reshape(-1,)
y_test = y_test.reshape(-1,)

print("✅ Data normalized successfully!")

# ===========================================
# 5️⃣ ANN Model Definition
# ===========================================
ann_model = models.Sequential([
    layers.Flatten(input_shape=(32, 32, 3)),
    layers.Dense(1000, activation='relu'),
    layers.Dense(1000, activation='relu'),
    layers.Dense(10, activation='softmax')
])

ann_model.compile(optimizer='sgd',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

print("\n🧠 ANN Model Summary:")
ann_model.summary()

# ===========================================
# 6️⃣ Train ANN
# ===========================================
ann_history = ann_model.fit(
    x_train, y_train,
    validation_data=(x_test, y_test),
    epochs=5,
    batch_size=64,
    verbose=2
)

# ===========================================
# 7️⃣ Evaluate ANN
# ===========================================
ann_loss, ann_acc = ann_model.evaluate(x_test, y_test, verbose=0)
print(f"✅ ANN Test Accuracy: {ann_acc:.3f}")

# ===========================================
# 8️⃣ CNN Model Definition
# ===========================================
cnn_model = models.Sequential([
    tf.keras.Input(shape=(32, 32, 3)),
    layers.Conv2D(32, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),

    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),

    layers.Conv2D(64, (3, 3), activation='relu'),

    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax')
])

cnn_model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

print("\n🧠 CNN Model Summary:")
cnn_model.summary()

# ===========================================
# 9️⃣ Train CNN
# ===========================================
cnn_history = cnn_model.fit(
    x_train, y_train,
    validation_data=(x_test, y_test),
    epochs=10,
    batch_size=64,
    verbose=2
)

# ===========================================
# 🔟 Evaluate CNN
# ===========================================
test_loss, test_acc = cnn_model.evaluate(x_test, y_test, verbose=0)
print(f"✅ CNN Test Accuracy: {test_acc:.3f}")

# ===========================================
# 1️⃣1️⃣ Plot Accuracy and Loss Curves
# ===========================================
plt.figure(figsize=(12, 4))

# Accuracy plot
plt.subplot(1, 2, 1)
plt.plot(cnn_history.history['accuracy'], label='Train Accuracy')
plt.plot(cnn_history.history['val_accuracy'], label='Validation Accuracy')
plt.title('CNN Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

# Loss plot
plt.subplot(1, 2, 2)
plt.plot(cnn_history.history['loss'], label='Train Loss')
plt.plot(cnn_history.history['val_loss'], label='Validation Loss')
plt.title('CNN Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.show()

# ===========================================
# 1️⃣2️⃣ Test a Random Image Prediction
# ===========================================
n = random.randint(0, len(x_test) - 1)
plt.imshow(x_test[n])
plt.title(f"True Label: {class_names[y_test[n]]}")
plt.axis('off')
plt.show()

pred = cnn_model.predict(x_test[n].reshape(1, 32, 32, 3))
print("🔍 Predicted Label:", class_names[np.argmax(pred)])

# ===========================================
# 1️⃣3️⃣ Save and Reload CNN Model
# ===========================================
cnn_model.save('cnn_cifar10_model.keras')
print("✅ Model saved successfully.")

reloaded_model = tf.keras.models.load_model('cnn_cifar10_model.keras')
print("✅ Model reloaded successfully.")
