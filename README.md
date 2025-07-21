# Fashion MNIST Classification using a Multilayer Perceptron (MLP) in TensorFlow / Keras

This project uses the [Fashion MNIST dataset](https://github.com/zalandoresearch/fashion-mnist) to classify clothing items using a simple Multilayer Perceptron (MLP) built with TensorFlow and Keras.

---

## 🧠 Model Overview

- **Architecture**: MLP with two hidden layers (default: 128 and 64 neurons, ReLU activation)
- **Output**: 10-class softmax classifier
- **Input**: 28x28 grayscale images, flattened to 784 features
- **Loss Function**: Sparse Categorical Crossentropy
- **Optimiser**: Adam
- **Metrics**: Accuracy

---

## 🧪 Performance (Default settings)

- **Epochs**: 10  
- **Batch size**: 32  
- **Validation Accuracy**: 0.8887  
- **Validation Loss**: 0.3240  
- **Test Accuracy**: 0.8775  
- **Test Loss**: 0.3409  

---

## 🔬 Experiments & Interpretations

### 🔁 Changing the Number of Epochs

| Epochs | Val Accuracy | Val Loss | Test Accuracy  | Test Loss  |
|--------|--------------|----------|----------------|------------|
| 5      | 0.8802       | 0.3246   | 0.8748         | 0.3447     |
| 10     | 0.8887       | 0.3240   | 0.8775         | 0.3409     |
| 20     | 0.8823       | 0.4026   | 0.8792         | 0.4156     |

**Interpretation**:
- Reducing epochs to 5 leads to underfitting (lower accuracy, higher loss) as the data is being 'seen' by the model less times and, therefore, losing some of the necessary refinements that are made with a higher number of epochs.
- Increasing to 20 causes overfitting (training improves, but generalisation worsens), where the data is being trained too well on the training set and, thus, makes errors on the validation and test sets.
- **Epoch = 10** appears to be the best balance.

---

### 🧱 Adjusting the Number of Neurons in Hidden Layers

| Neurons (1st layer) | Val Accuracy | Val Loss | Test Accuracy  | Test Loss  |
|---------------------|--------------|----------|----------------|------------|
| 128 (default)       | 0.8887       | 0.3240   | 0.8775         | 0.3409     |
| 256                 | 0.8925       | 0.3160   | 0.8818         | 0.3439     |

**Interpretation**:
- More neurons (**256**) increased accuracy slightly.
- Larger networks may improve performance, but at the cost of:
  - Increased compute/memory usage.
  - Risk of overfitting.
  - Potential optimisation issues (e.g., dying ReLU here, or vanishing gradient if using the sigmoid function).

---

### 📦 Changing the Batch Size

| Batch Size | Val Accuracy | Val Loss | Test Accuracy  | Test Loss  |
|------------|--------------|----------|----------------|------------|
| 32         | 0.8887       | 0.3240   | 0.8775         | 0.3409     |
| 64         | 0.8888       | 0.3076   | 0.8815         | 0.3295     |
| 128        | 0.8830       | 0.3311   | 0.8794         | 0.3372     |

**Interpretation**:
- Larger batch sizes offer faster training and smoother gradients but may settle in poorer local minima.
- Smaller batches can generalise better by adding noise to the gradient estimation.
- A batch size of **64**, therefore, is a good balance between gradient stability and the exploration of the loss landscape, allowing the model to converge to a better set of weights that generalise well.

---


## 📸 Sample Prediction Output
```
Sample 1: Predicted: Ankle boot, Actual: Ankle boot
Sample 2: Predicted: Pullover, Actual: Pullover
Sample 3: Predicted: Trouser, Actual: Trouser
Sample 4: Predicted: Trouser, Actual: Trouser
Sample 5: Predicted: Shirt, Actual: Shirt
```

---

## 📂 Dataset Info

Fashion MNIST is a dataset of Zalando's article images:
- 60,000 training images
- 10,000 test images
- 28x28 grayscale images
- 10 categories of clothing

---

## 🧑‍💻 Author

This repo is part of a broader portfolio on learning machine learning and deep learning from the ground up.
