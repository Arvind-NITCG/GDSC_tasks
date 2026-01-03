# 👁️ Handwritten Digit Recognition: A CNN Approach

### **Author:** Arvind K N
**Project Context:** Intermediate Level Task (GDSC Induction).
**Goal:** To build a Deep Learning model capable of recognizing handwritten digits (0-9) from the MNIST dataset with high precision (>99%).

---

## 1. The Problem: Why not simple Neural Networks? 🤔
An image is a grid of pixels (28x28 = 784 pixels).
If we used a standard Dense Neural Network (like in the Placement Prediction task), we would have to flatten the image into a long row of 784 numbers immediately.

**The Flaw:** Flattening destroys **Spatial Structure**.
* In a line, the pixel at `(0,0)` is far away from `(0,1)`, but in an image, they are neighbors.
* A standard neural network doesn't know that a "circle" shape at the top-left is the same as a "circle" at the bottom-right.

**The Solution:** **Convolutional Neural Networks (CNNs)**.
Instead of looking at all pixels at once, we scan the image with small "windows" (filters) to detect patterns (edges, curves, loops) regardless of where they are.

---

## 2. Model Architecture: "The Eyes and The Brain" 🧠

The model architecture is split into two distinct parts:

### Phase 1: Feature Extraction ("The Eyes") 📷
This part looks at the image and extracts meaningful features.

1.  **Conv2D Layer (32 Filters, 3x3 Kernel):**
    * **Math:** The core operation is the **Convolution Integral** (discrete version):
        $$S(i, j) = (I * K)(i, j) = \sum_m \sum_n I(i+m, j+n) \cdot K(m, n)$$
    * **Logic:** We slide 32 different small $3\times3$ matrices (kernels) over the image. One kernel might look for vertical lines, another for horizontal lines.
    * *Activation:* **ReLU** ($max(0, x)$) is used to introduce non-linearity (turning off negative correlations).

2.  **MaxPooling2D (2x2):**
    * **Math:** Downsampling.
        $$Output_{h,w} = \max(Window_{2\times2})$$
    * **Logic:** We shrink the image by half. If we found a "curve" in a 2x2 region, we just keep the strongest signal. This reduces computation and makes the model **translation invariant** (it doesn't matter if the digit shifts slightly).

3.  **Conv2D Layer (64 Filters, 3x3 Kernel):**
    * Now that the image is smaller, we look for more complex features (combinations of lines, like corners or loops) using 64 filters.

### Phase 2: Classification ("The Brain") ⚡
Now that we have a map of "where the loops and lines are," we decide which digit it is.

4.  **Flatten:** Unroll the 3D feature maps into a 1D vector.
5.  **Dense (64, ReLU):** A standard fully connected layer to interpret the features.
6.  **Dense (10, Softmax):** The Output Layer.
    * **Math:** Converts raw scores ($logits$) into probabilities.
        $$P(y=k) = \frac{e^{z_k}}{\sum_{j=0}^{9} e^{z_j}}$$
    * The neuron with the highest probability is our predicted digit.

---

## 3. Training Dynamics ⚙️

* **Optimizer:** `Adam` (Adaptive Moment Estimation). It adjusts the learning rate for each weight individually, converging faster than standard SGD.
* **Loss Function:** `sparse_categorical_crossentropy`.
    $$Loss = - \sum_{i} y_{true} \cdot \log(y_{pred})$$
    * This penalizes the model heavily if it assigns a low probability to the correct digit.
* **Normalization:**
    * Input pixels are divided by **255.0**. Neural Networks converge faster when inputs are between 0 and 1 (Gradient Descent moves smoothly).

---

## 4. Results 🏆
* **Dataset:** MNIST (60,000 Training, 10,000 Test images).
* **Test Accuracy:** **~99.0%**
* The model successfully learned to distinguish similar digits (like 1 vs 7, or 3 vs 8) by learning their unique geometric features.

---

## 5. How to Run 🚀
1.  **Install Dependencies:**
    ```bash
    pip install tensorflow matplotlib
    ```
2.  **Run the Notebook:**
    Execute `Task_intermidiate.ipynb`. The script will automatically download the dataset and start training.

---

*"Computer Vision is not just about seeing, it's about understanding geometry."*