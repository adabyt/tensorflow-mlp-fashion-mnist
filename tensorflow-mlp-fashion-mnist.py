import tensorflow as tf
from tensorflow import keras
from keras import layers 
from keras import models 

import numpy as np
import matplotlib.pyplot as plt 

print("----- 1. Loading and Initial Data Preparation -----")

# Load the Fashion MNIST dataset
# x_train, x_test are images (features)
# y_train, y_test are labels (targets)
(x_train, y_train), (x_test, y_test) = keras.datasets.fashion_mnist.load_data()

print(f"Original x_train shape: {x_train.shape}")   # (60000, 28, 28)
print(f"Original y_train shape: {y_train.shape}")   # (60000,)
print(f"Original x_test shape: {x_test.shape}")     # (10000, 28, 28)
print(f"Original y_test shape: {y_test.shape}")     # (10000,)

# Let's look at one sample image
plt.imshow(x_train[0], cmap='gray')
plt.title(f"Label: {y_train[0]}")
plt.show()

# Define class names for better understanding
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
print(f"Example label (numerical): {y_train[0]}")           # 9         
print(f"Example label (text): {class_names[y_train[0]]}")   # # Ankle boot

print("-"*100)

print("----- 2. Data Preprocessing for MLP -----")

"""
Before feeding the data to our MLP, we need to perform two key preprocessing steps:
- Normalisation: Pixel values in images typically range from 0 to 255. We need to scale these to a smaller range, usually 0 to 1, for better network training.
- Flattening: An MLP expects a 1-D array of features for each sample. Our images are 2D (28x28). We need to "flatten" each 28x28 image into a single 1D array of 28 * 28 = 784 features.
"""

# Convert integer pixel values to float32 for calculations (from uint8)
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

# Normalise pixel values to be between 0 and 1
# Original pixel values are 0-255. Divide by 255.0 to scale.
x_train = x_train / 255.0
x_test = x_test / 255.0

print(f"\nx_train shape after dtype conversion and normalization: {x_train.shape}") # (60000, 28, 28)
print(f"x_test shape after dtype conversion and normalization: {x_test.shape}")     # (10000, 28, 28)

# Flatten the images: (60000, 28, 28) -> (60000, 784)
# -1 in reshape retains the number of samples (rows in the first dimension) as they are (i.e. 60000)
x_train_flat = x_train.reshape(-1, 28 * 28)
x_test_flat = x_test.reshape(-1, 28 * 28)
# Could also use the following if the dimensions were not known:
# x_train_flat = x_train.reshape(-1, x_train.shape[1] * x_train.shape[2])
# x_test_flat = x_test.reshape(-1, x_test.shape[1] * x_test.shape[2])

print(f"x_train_flat shape after flattening: {x_train_flat.shape}")                 # (60000, 784)
print(f"x_test_flat shape after flattening: {x_test_flat.shape}")                   # (10000, 784)

print("-"*100)

print("----- 3. Model Definition -----")

# Define the Keras Sequential model
model = models.Sequential([
    # Input Layer (implicitly defined by the first Dense layer's input_shape)
    # This is our first hidden layer (Dense layer 1)
    layers.Dense(128, activation='relu', input_shape=(784,)),   # 784 features as input (28 * 28)
    
    # Second hidden layer (Dense layer 2)
    layers.Dense(64, activation='relu'),
    
    # Output Layer
    # 10 neurons for 10 classes (0-9)
    # 'softmax' activation for multi-class classification
    # Softmax converts raw scores (logits) into probabilities that sum to 1.
    layers.Dense(10, activation='softmax')
])

model.summary()
# Model: "sequential"
# ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━┓
# ┃ Layer (type)                         ┃ Output Shape                ┃         Param # ┃
# ┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━┩
# │ dense (Dense)                        │ (None, 128)                 │         100,480 │  # (784 * 128) + 128 (inputs * weights + bias)
# ├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
# │ dense_1 (Dense)                      │ (None, 64)                  │           8,256 │
# ├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
# │ dense_2 (Dense)                      │ (None, 10)                  │             650 │
# └──────────────────────────────────────┴─────────────────────────────┴─────────────────┘
#  Total params: 109,386 (427.29 KB)
#  Trainable params: 109,386 (427.29 KB)
#  Non-trainable params: 0 (0.00 B)

print("-"*100)

print("----- 4. Model Compilation -----")

# Compile the model
model.compile(optimizer='adam',                                                     # Our optimiser Adam (Adaptive Moment Estimation)
              loss=keras.losses.SparseCategoricalCrossentropy(from_logits=False),   # Loss function
              metrics=['accuracy'])                                                 # Metrics to monitor


print("-"*100)

print("----- 5. Model Training -----")

# Train the model
# x_train_flat: our flattened training images (features)
# y_train: our integer training labels (targets)
# epochs: how many times the model will see the entire training dataset
# batch_size: how many samples to process before updating weights (mini-batch SGD)
    # Gradient Descent (Full Batch GD): Calculates the gradient of the loss function using all the training examples
    # Stochastic Gradient Descent: Calculates the gradient using only one randomly chosen training example at a time
    # Mini-Batch SGD: Instead of using all data or just one sample, calculate the gradient and update weights using a small "batch" of training examples (e.g., 32, 64, 128 samples
# validation_split: percentage of training data to use for validation during training
history = model.fit(x_train_flat, y_train,
                    epochs=10,              # Adjust as needed
                    batch_size=32,          # Common batch size (Number of batches per epoch = 60,000 / 32 = 1,875 batches)
                    validation_split=0.1)   # Use 10% of training data for validation
# For this model: 
    # Total weight updates = epochs * (number of batches per epoch) = 10 * 1,875 = 18,750 weight updates over the entire training process

# Epoch 1/10
# 1688/1688 ━━━━━━━━━━━━━━━━━━━━ 1s 655us/step - accuracy: 0.7733 - loss: 0.6472 - val_accuracy: 0.8558 - val_loss: 0.3917
# Epoch 2/10
# 1688/1688 ━━━━━━━━━━━━━━━━━━━━ 1s 627us/step - accuracy: 0.8569 - loss: 0.3875 - val_accuracy: 0.8702 - val_loss: 0.3571
# Epoch 3/10
# 1688/1688 ━━━━━━━━━━━━━━━━━━━━ 1s 621us/step - accuracy: 0.8751 - loss: 0.3416 - val_accuracy: 0.8705 - val_loss: 0.3573
# Epoch 4/10
# 1688/1688 ━━━━━━━━━━━━━━━━━━━━ 1s 625us/step - accuracy: 0.8847 - loss: 0.3118 - val_accuracy: 0.8718 - val_loss: 0.3563
# Epoch 5/10
# 1688/1688 ━━━━━━━━━━━━━━━━━━━━ 1s 636us/step - accuracy: 0.8888 - loss: 0.2994 - val_accuracy: 0.8837 - val_loss: 0.3316
# Epoch 6/10
# 1688/1688 ━━━━━━━━━━━━━━━━━━━━ 1s 636us/step - accuracy: 0.8966 - loss: 0.2769 - val_accuracy: 0.8802 - val_loss: 0.3243
# Epoch 7/10
# 1688/1688 ━━━━━━━━━━━━━━━━━━━━ 1s 634us/step - accuracy: 0.9021 - loss: 0.2629 - val_accuracy: 0.8782 - val_loss: 0.3368
# Epoch 8/10
# 1688/1688 ━━━━━━━━━━━━━━━━━━━━ 1s 625us/step - accuracy: 0.9056 - loss: 0.2527 - val_accuracy: 0.8890 - val_loss: 0.3138  # Potential overfitting after this point
# Epoch 9/10
# 1688/1688 ━━━━━━━━━━━━━━━━━━━━ 1s 639us/step - accuracy: 0.9090 - loss: 0.2451 - val_accuracy: 0.8763 - val_loss: 0.3377
# Epoch 10/10
# 1688/1688 ━━━━━━━━━━━━━━━━━━━━ 1s 639us/step - accuracy: 0.9118 - loss: 0.2365 - val_accuracy: 0.8887 - val_loss: 0.3240


print("-"*100)

print("----- 6. Model Evaluation -----")

# Evaluate the model on the test data
test_loss, test_accuracy = model.evaluate(x_test_flat, y_test, verbose=1)

print(f"\nTest Loss: {test_loss:.4f}")          # 0.3409
print(f"Test Accuracy: {test_accuracy:.4f}")    # 0.8775

print("-"*100)

print("----- 7. Making Predictions -----")

# Make predictions on a few test samples (i.e. new, unseen data)
predictions = model.predict(x_test_flat[:5]) # Predict on the first 5 test images

print(f"\nPredictions for the first 5 test samples:\n{predictions}")

# The predictions are probabilities. To get the actual predicted class, use argmax.
predicted_classes = np.argmax(predictions, axis=1)

print(f"\nPredicted classes for the first 5 samples: {predicted_classes}")  # [9 2 1 1 6]
print(f"Actual classes for the first 5 samples: {y_test[:5]}")              # [9 2 1 1 6]

# Let's see the actual class names for comparison
print("\nComparison of predicted vs actual:")
for i in range(5):
    print(f"Sample {i+1}: Predicted: {class_names[predicted_classes[i]]} (Index: {predicted_classes[i]}), Actual: {class_names[y_test[i]]} (Index: {y_test[i]})")

# Comparison of predicted vs actual:
# Sample 1: Predicted: Ankle boot (Index: 9), Actual: Ankle boot (Index: 9)
# Sample 2: Predicted: Pullover (Index: 2), Actual: Pullover (Index: 2)
# Sample 3: Predicted: Trouser (Index: 1), Actual: Trouser (Index: 1)
# Sample 4: Predicted: Trouser (Index: 1), Actual: Trouser (Index: 1)
# Sample 5: Predicted: Shirt (Index: 6), Actual: Shirt (Index: 6)
