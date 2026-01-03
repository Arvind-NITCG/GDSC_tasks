# GPT-2 Architecture from Scratch: A Mathematical Deep Dive 🧠✨

### **Author:** Arvind K N
**Project Context:** A first-principles implementation of the Transformer Decoder (GPT-2 variant) using PyTorch, demonstrating the underlying tensor calculus and linear algebra.

---

## 1. Project Overview 🏗️
This repository contains a faithful reconstruction of the **GPT-2 Decoder-Only Architecture**. Unlike high-level API implementations, this project explicitly defines the matrix operations, broadcasting mechanics, and attention mathematics that enable Large Language Models to function.

The model is **autoregressive**: it predicts the next token $x_{t+1}$ based on the history of tokens $x_{1}, ..., x_{t}$.

---

## 2. Mathematical Architecture Breakdown

### 2.1. Input Processing: Embeddings & Position 📥
The model cannot process raw text. We convert tokens into continuous vector space.

* **Input Tensor ($X$):** Shape `[Batch, Seq_Len]`.
* **Token Embedding:** Maps integer IDs to vectors of size $d_{model}$ (512).
    $$E_{token} = \text{EmbeddingTable}[X]$$
* **Positional Encoding ($PE$):** Since Self-Attention is permutation invariant, we inject geometric order using fixed sinusoidal waves.
    $$PE_{(pos, 2i)} = \sin\left(\frac{pos}{10000^{2i/d_{model}}}\right)$$
    $$PE_{(pos, 2i+1)} = \cos\left(\frac{pos}{10000^{2i/d_{model}}}\right)$$
* **Final Input:** $X_{input} = E_{token} + PE$

---

### 2.2. The Decoder Block (Repeated 12x) 🔄
The core computation happens in a stack of 12 identical layers. Each layer consists of two sub-layers: **Masked Multi-Head Attention** and a **Feed-Forward Network**, connected by **Residuals**.

#### A. Masked Multi-Head Self-Attention
We project the input $X$ into three subspaces to calculate relevance.

1.  **Linear Projections ($Q, K, V$):**
    $$Q = XW^Q, \quad K = XW^K, \quad V = XW^V$$
    * *Dimensions:* $[Batch, Seq, 512] \times [512, 512] \rightarrow [Batch, Seq, 512]$

2.  **Head Splitting (Parallelization):**
    We split the 512 dimension into $h=8$ heads of size $d_k=64$.
    * *Transformation:* Reshape & Transpose to $[Batch, Heads, Seq, d_k]$.

3.  **Scaled Dot-Product Attention (The Core Math):**
    $$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}} + M\right)V$$
    * **$QK^T$:** Computes similarity scores between all pairs of tokens. Shape: $[Batch, Heads, Seq, Seq]$.
    * **Scaling ($\sqrt{d_k}$):** Stabilizes gradients by keeping dot products small.
    * **Causal Mask ($M$):** A lower-triangular matrix (upper values $= -\infty$). This enforces the physical constraint that **current tokens cannot see future tokens**.
    * **Softmax:** Normalizes scores along the last dimension so they sum to 1.0.

4.  **Output Projection:**
    The heads are concatenated back to $[Batch, Seq, 512]$ and mixed via a final linear layer $W^O$.

#### B. Position-wise Feed-Forward Network (FFN)
A simple MLP applied to every token independently (Key-Value memory).
$$FFN(x) = \text{ReLU}(xW_1 + b_1)W_2 + b_2$$
* **Expansion:** Project $512 \rightarrow 2048$.
* **Contraction:** Project $2048 \rightarrow 512$.

#### C. Residual Connections & Layer Normalization 🛡️
Crucial for deep network training (vanishing gradient protection).
We employ **Add & Norm** around each sub-layer:

$$x_{out} = \text{LayerNorm}(x_{in} + \text{SubLayer}(x_{in}))$$

* **Residual ($+$):** The input is added directly to the output. This creates a "gradient superhighway" allowing gradients to flow through the network unchanged.
* **LayerNorm:** Normalizes the vector statistics (Mean=0, Var=1) for stability.
    $$LN(x) = \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}} \cdot \gamma + \beta$$

---

### 2.3. The Output Layer (Prediction Head) 🎯
After 12 layers of processing, we project the rich latent representation back to the vocabulary space.

1.  **Final Normalization:** One last `LayerNorm` is applied to stabilize the final features.
2.  **Linear Projection (Un-embedding):**
    $$Logits = X_{final} \cdot W_{vocab}^T$$
    * *Dimensions:* $[Batch, Seq, 512] \times [512, Vocab\_Size] \rightarrow [Batch, Seq, Vocab\_Size]$.
    * This produces a score for every word in the dictionary (e.g., 50,000 words) for every position in the sequence.
3.  **Softmax (Probability Distribution):**
    We convert logits into probabilities to predict the next word.
    $$P(x_{t+1}) = \text{Softmax}(Logits_t)$$

---

## 3. Tensor Flow Summary 📊

| Step | Tensor Shape | Operation Meaning |
| :--- | :--- | :--- |
| **Input** | `[B, Seq]` | Raw Token IDs |
| **Embedding** | `[B, Seq, 512]` | Vector Representation + Time Signal |
| **Q/K/V Proj** | `[B, Seq, 512]` | Preparation for Attention |
| **Attention Scores** | `[B, 8, Seq, Seq]` | "Who is looking at whom?" |
| **Context Vector** | `[B, Seq, 512]` | Weighted sum of relevant history |
| **FFN Expansion** | `[B, Seq, 2048]` | Processing/Reasoning phase |
| **Logits** | `[B, Seq, Vocab]` | Raw prediction scores |
| **Output** | `[B, Seq, Vocab]` | Probability distribution over next token |

---

## 4. Why From Scratch?
Implementing `torch.nn.MultiheadAttention` is easy. Implementing the matrix calculus `(Q @ K.T) / sqrt(d_k)` manually proves understanding of:
1.  **Broadcasting rules** in high-dimensional tensors.
2.  **Manifold geometry** (Positional Encoding).
3.  **Optimization dynamics** (Residuals & Normalization).

---
