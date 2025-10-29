# RoFormer: Enhanced Transformer with Rotary Position Embedding

**Presenter:** Charlee Kraiss  
**Course:** DS 5690-01 Gen AI Models in Theory and Practice (2025F)  
**Date:** 30 - Oct - 2025

---

## Paper Information

**Title:** RoFormer: Enhanced Transformer with Rotary Position Embedding

**Authors:** Jianlin Su, Yu Lu, Shengfeng Pan, Ahmed Murtadha, Bo Wen, Yunfeng Liu (Zhuiyi Technology Co., Ltd.)

**Published:** November 9, 2023 (arXiv:2104.09864v5)

**Citation:**
```
Su, J., Lu, Y., Pan, S., Murtadha, A., Wen, B., & Liu, Y. (2023). 
RoFormer: Enhanced Transformer with Rotary Position Embedding. 
arXiv preprint arXiv:2104.09864v5.
```

**Paper Link:** https://arxiv.org/abs/2104.09864

---

## The Problem: Position-Agnostic Transformers

### Transformers Don't Understand Order

Without positional information, transformers treat sequences as "bags of words."

**Example:** 
- "the cat chased the mouse" 
- "the mouse chased the cat"

**Same tokens → identical representations**, but completely different meanings!

The self-attention mechanism from our Formal Algorithms paper (Algorithm 4) computes:

$$q_m^T k_n = (W_q x_m)^T (W_k x_n)$$

This is **position-agnostic** - the inner product doesn't know that token $m$ comes before token $n$.

### Previous Solution: Additive Position Encoding

From Formal Algorithms (Algorithm 2), traditional transformers **add** position embeddings:

**Algorithm: Traditional Position Encoding**

**Input:** $v \in V$, token ID; $t \in [\ell_{\max}]$, position

**Output:** $e \in \mathbb{R}^{d_e}$, embedded token with position

**Parameters:** 
- $W_e \in \mathbb{R}^{d_e \times N_V}$, token embedding matrix
- $W_p \in \mathbb{R}^{d_e \times \ell_{\max}}$, position embedding matrix

**Algorithm:**
1. $e \leftarrow W_e[:, v] + W_p[:, t]$ $\triangleright$ Add position to token
2. **return** $e$

This combined embedding then goes into attention (Algorithm 4).

### Problems with Additive Encoding

1. **Mixed representations:** Position and content information get blended in embedding space
2. **No explicit relative encoding:** Attention must learn to extract relative distances from absolute positions
3. **Fixed maximum length:** $W_p$ has fixed dimensions for $\ell_{\max}$ - can't handle longer sequences
4. **Breaks linear attention:** Incompatible with $O(N)$ attention mechanisms

---

## The Solution: Rotary Position Embedding (RoPE)

### Core Insight

Instead of **adding** position to embeddings, **rotate** query and key vectors during attention computation.

**Key mathematical property:**
$$\text{Rotation by angle } m\theta \text{ then by } n\theta = \text{Rotation by } (n-m)\theta$$

This naturally encodes **relative distance** $(n-m)$, not absolute positions!

### How RoPE Works

**Traditional Attention:**
1. Add position to embedding: $x + p_m$
2. Compute Q, K: $W_q(x + p_m)$, $W_k(x + p_n)$
3. Attention depends on both position and content mixed together

**RoPE Attention:**
1. Keep embeddings pure: $x$ (no position added)
2. Compute Q, K: $W_q x$, $W_k x$
3. **Rotate** by position: $R_m(W_q x)$, $R_n(W_k x)$
4. Attention naturally depends on relative distance $(n-m)$

Position information enters **during** attention computation, not **before** it!

---

## Architecture: RoPE Attention Algorithm

Modified from Formal Algorithms Algorithm 4:

**Input:** 
- $X \in \mathbb{R}^{d_x \times \ell_x}$, $Z \in \mathbb{R}^{d_z \times \ell_z}$, token sequences

**Output:** 
- $\tilde{X} \in \mathbb{R}^{d_{\text{out}} \times \ell_x}$, updated representations

**Parameters:** 
- $W_q \in \mathbb{R}^{d_{\text{attn}} \times d_x}$ (no $b_q$ with RoPE)
- $W_k \in \mathbb{R}^{d_{\text{attn}} \times d_z}$ (no $b_k$ with RoPE)
- $W_v \in \mathbb{R}^{d_{\text{out}} \times d_z}$, $b_v \in \mathbb{R}^{d_{\text{out}}}$

**Hyperparameters:** 
- $\Theta = \{\theta_i = 10000^{-2(i-1)/d} : i \in [d/2]\}$

**Algorithm:**
1. **for** $m \in [\ell_x]$ **do** $\triangleright$ Construct rotation matrix for each position $m$

The full rotation matrix for position $m$ in $d$ dimensions is a block-diagonal matrix:

$$
R^d_{\Theta,m} = 
\begin{bmatrix}
\cos(m\theta_1) & -\sin(m\theta_1) & 0 & 0 & \cdots & 0 & 0 \\
\sin(m\theta_1) & \cos(m\theta_1) & 0 & 0 & \cdots & 0 & 0 \\
0 & 0 & \cos(m\theta_2) & -\sin(m\theta_2) & \cdots & 0 & 0 \\
0 & 0 & \sin(m\theta_2) & \cos(m\theta_2) & \cdots & 0 & 0 \\
\vdots & \vdots & \vdots & \vdots & \ddots & \vdots & \vdots \\
0 & 0 & 0 & 0 & \cdots & \cos(m\theta_{d/2}) & -\sin(m\theta_{d/2}) \\
0 & 0 & 0 & 0 & \cdots & \sin(m\theta_{d/2}) & \cos(m\theta_{d/2})
\end{bmatrix}
$$

Each 2×2 block along the diagonal is a rotation matrix:

$$
R(m\theta_i) = 
\begin{bmatrix}
\cos(m\theta_i) & -\sin(m\theta_i) \\
\sin(m\theta_i) & \cos(m\theta_i)
\end{bmatrix}
$$

The block-diagonal structure means each pair of dimensions $(2i-1, 2i)$ is rotated independently by angle $m\theta_i$.

2. **end for**
3. $Q \leftarrow [R^d_{\Theta,1} W_q X[:,1], R^d_{\Theta,2} W_q X[:,2], \ldots, R^d_{\Theta,\ell_x} W_q X[:,\ell_x]]$ $\triangleright$ Apply rotation to queries and concatenate

**How Q is Built Column by Column:**

$$
Q = \begin{bmatrix} 
| & | & | & | & | \\ 
R_1(W_q X[:,1]) & R_2(W_q X[:,2]) & R_3(W_q X[:,3]) & R_4(W_q X[:,4]) & R_5(W_q X[:,5]) \\ 
| & | & | & | & | 
\end{bmatrix}
$$

4. $K \leftarrow [R^d_{\Theta,1} W_k Z[:,1], R^d_{\Theta,2} W_k Z[:,2], \ldots, R^d_{\Theta,\ell_z} W_k Z[:,\ell_z]]$ $\triangleright$ Apply rotation to keys
5. $V \leftarrow W_v Z + b_v \mathbf{1}^T$ $\triangleright$ Values NOT rotated
6. $S \leftarrow Q^T K$ $\triangleright$ Compute attention scores *Relative Position Magic Here!*
7. Apply masking to $S$ if needed
8. **return** $\tilde{X} \leftarrow V \cdot \text{softmax}(S / \sqrt{d_{\text{attn}}})$

**Key differences from Formal Algorithms Algorithm 4:**
- **Line 3-4:** Instead of $Q \leftarrow W_q X + b_q \mathbf{1}^T$, we rotate: $Q \leftarrow [R^d_{\Theta,m} W_q X[:,m]]$
- **No bias terms:** RoPE omits bias for Q and K projections ($b_q$ and $b_k$) because rotating a bias makes it position-dependent ($R_m b_q$ varies with $m$), which breaks the relative position property. The value projection can still use bias since it's not rotated.
- No separate position embedding step (no Algorithm 2 equivalent)
- Position information appears during attention computation, not in embeddings

---

## Understanding θ_i: Multiple Rotation Frequencies

In our concrete example below, we'll use **$d = 4$ dimensions with two rotation frequencies** to demonstrate how multiple frequencies work. For production models with higher-dimensional embeddings (e.g., $d = 512$ or $d = 768$), RoPE uses even more rotation frequencies - one $\theta_i$ for each dimension pair.

**The formula:**
$$\theta_i = 10000^{-2i/d} \quad \text{for } i = 1, 2, \ldots, d/2$$

This creates a geometric series:
- $\theta_1 = 10000^{-2/d}$ → **slow rotation** (small angle per position)
- $\theta_2 = 10000^{-4/d}$ → **medium rotation**
- $\theta_{d/2} = 10000^{-1} = 0.0001$ → **fast rotation** (large angle per position)

**Why multiple frequencies?**

Different frequencies capture positional relationships at different scales:

| Frequency | Rotation Speed | What It Captures |
|-----------|---------------|------------------|
| Low ($\theta_1$) | Slow, takes many positions to rotate significantly | **Long-range dependencies** - "this word is 50 tokens away" |
| High ($\theta_{d/2}$) | Fast, rotates significantly between adjacent positions | **Local dependencies** - "these words are adjacent" |

This is analogous to sinusoidal position embeddings having both low and high frequency components.

**For our 2D example:** With $d=2$, we have only one dimension pair, so $i=1$ and:
$$\theta_1 = 10000^{-2/2} = 10000^{-1} = 0.0001$$

We simplified to $\theta = 1.0$ for illustration, but the actual RoFormer implementation would use $\theta_1 = 0.0001$.

**Extending to 4D:**
If we had $d=4$, we'd have two rotation frequencies:
- Dimensions 1-2: rotated by $\theta_1 = 10000^{-2/4} = 0.01$
- Dimensions 3-4: rotated by $\theta_2 = 10000^{-4/4} = 0.0001$

The first pair rotates more slowly (captures longer-range patterns), while the second rotates faster (captures local patterns).

---

## Concrete Example: "The Cat Chased The Mouse"

Let's walk through **"the cat chased the mouse"** with actual numbers to see exactly how RoPE works with multiple rotation frequencies.

### Setup
- **Sentence:** "the cat chased the mouse"
- **Token positions:** 
  - "the" = position 1
  - "cat" = position 2  
  - "chased" = position 3
  - "the" = position 4
  - "mouse" = position 5
- **Dimensions:** $d = 4$ (instead of 768+, but showing the multi-frequency concept)
- **Rotation frequencies:** 
  - $\theta_1 = 10000^{-2/4} = 0.01$ (slow rotation for dimensions 1-2)
  - $\theta_2 = 10000^{-4/4} = 0.0001$ (fast rotation for dimensions 3-4)

### Step 1: Token Embeddings (Before Any Position Information)

Imagine these are learned 4D embeddings for each word:

$$
X = 
\begin{bmatrix}
0.5 & 0.9 & 0.2 & 0.5 & 0.3 \\
0.3 & 0.4 & 0.8 & 0.3 & 0.7 \\
0.6 & 0.6 & 0.5 & 0.4 & 0.4 \\
0.2 & 0.3 & 0.7 & 0.6 & 0.8
\end{bmatrix}
$$

Where columns represent: ["the", "cat", "chased", "the", "mouse"]

**Note:** These embeddings contain NO position information yet! Each token is a 4D point in embedding space.

### Step 2: Apply Linear Transformations (Still No Position)

Using simplified $W_q = W_k = I$ (identity matrix) for clarity:

$$Q = W_q X = X$$
$$K = W_k X = X$$

In practice, $W_q$ and $W_k$ are learned weight matrices.

### Step 3: Rotate queries and keys by their position

**Using d = 4 dimensions with two rotation frequencies:**

Recall that different dimension pairs rotate at different speeds:
- **Pair 1** (dimensions 1-2): Rotates by $m\theta_1 = m \times 0.01$ radians
- **Pair 2** (dimensions 3-4): Rotates by $m\theta_2 = m \times 0.0001$ radians

**For "cat" at position $m = 2$:**

Rotation angles:
- Dimensions 1-2: $2 \times 0.01 = 0.02$ radians
- Dimensions 3-4: $2 \times 0.0001 = 0.0002$ radians

Rotation matrix $R_2$ (4×4 block-diagonal):

$$
R_2 = \begin{bmatrix}
\cos(0.02) & -\sin(0.02) & 0 & 0 \\
\sin(0.02) & \cos(0.02) & 0 & 0 \\
0 & 0 & \cos(0.0002) & -\sin(0.0002) \\
0 & 0 & \sin(0.0002) & \cos(0.0002)
\end{bmatrix} \approx \begin{bmatrix}
0.9998 & -0.0200 & 0 & 0 \\
0.0200 & 0.9998 & 0 & 0 \\
0 & 0 & 1.0000 & -0.0002 \\
0 & 0 & 0.0002 & 1.0000
\end{bmatrix}
$$

**Notice:** 
- **First 2×2 block** (slow $\theta_1$): Small but visible rotation (≈1.1°)
- **Second 2×2 block** (fast $\theta_2$): Barely rotated at all! (≈0.01°)

Rotated query for "cat":

$$
q_2 = R_2 \begin{bmatrix} 0.9 \\ 0.4 \\ 0.6 \\ 0.3 \end{bmatrix} = \begin{bmatrix} 0.8918 \\ 0.4179 \\ 0.5999 \\ 0.3001 \end{bmatrix}
$$

**For "chased" at position $n = 3$:**

Rotation angles:
- Dimensions 1-2: $3 \times 0.01 = 0.03$ radians
- Dimensions 3-4: $3 \times 0.0001 = 0.0003$ radians

Rotation matrix $R_3$:

$$
R_3 \approx \begin{bmatrix}
0.9996 & -0.0300 & 0 & 0 \\
0.0300 & 0.9996 & 0 & 0 \\
0 & 0 & 1.0000 & -0.0003 \\
0 & 0 & 0.0003 & 1.0000
\end{bmatrix}
$$

Rotated key for "chased":

$$
k_3 = R_3 \begin{bmatrix} 0.2 \\ 0.8 \\ 0.5 \\ 0.7 \end{bmatrix} = \begin{bmatrix} 0.1759 \\ 0.8059 \\ 0.4998 \\ 0.7002 \end{bmatrix}
$$

**For "mouse" at position $n = 5$:**

Rotation angles:
- Dimensions 1-2: $5 \times 0.01 = 0.05$ radians  
- Dimensions 3-4: $5 \times 0.0001 = 0.0005$ radians

Rotation matrix $R_5$:

$$
R_5 \approx \begin{bmatrix}
0.9988 & -0.0500 & 0 & 0 \\
0.0500 & 0.9988 & 0 & 0 \\
0 & 0 & 1.0000 & -0.0005 \\
0 & 0 & 0.0005 & 1.0000
\end{bmatrix}
$$

**Notice:** 
- **First block**: More rotation than positions 2 or 3 (≈2.9°)
- **Second block**: Still barely rotated! (≈0.03°) Fast $\theta$ means slow angle growth

Rotated key for "mouse":

$$
k_5 = R_5 \begin{bmatrix} 0.3 \\ 0.7 \\ 0.4 \\ 0.8 \end{bmatrix} = \begin{bmatrix} 0.2646 \\ 0.7137 \\ 0.3996 \\ 0.8002 \end{bmatrix}
$$

### Visualization: Multi-Frequency Rotation

We can visualize each 2D subspace separately:

**Dimensions 1-2 (slow rotation, $\theta_1 = 0.01$):**
```
        North
            ↑
            |   • mouse (0.05 rad ≈ 2.9°)
            |  /
            | / • chased (0.03 rad ≈ 1.7°)
            |/ • cat (0.02 rad ≈ 1.1°)
West ←------O------→ East
            |
         Start
```
Small angles → all tokens still relatively close together in this subspace.

**Dimensions 3-4 (fast rotation, $\theta_2 = 0.0001$):**
```
        North
            ↑
            | • all tokens (0.0002-0.0005 rad)
            | clustered extremely close
            |
West ←------O------→ East
            |
         Start
```
Tiny angles → tokens barely rotated at all! These dimensions maintain similarity even for "distant" tokens at these positions.

**Key insight:** The two subspaces capture position information at different scales!

### Step 4: Compute Attention Scores

**"cat" → "chased" (adjacent, distance = 1):**

$$q_2^T k_3 = (0.8918)(0.1759) + (0.4179)(0.8059) + (0.5999)(0.4998) + (0.3001)(0.7002)$$

$$= 0.157 + 0.337 + 0.300 + 0.210 = 1.004$$

**"cat" → "mouse" (distant, distance = 3):**

$$q_2^T k_5 = (0.8918)(0.2646) + (0.4179)(0.7137) + (0.5999)(0.3996) + (0.3001)(0.8002)$$

$$= 0.236 + 0.298 + 0.240 + 0.240 = 1.014$$

### Key Observations About These Scores

**Both scores are positive and relatively high!** At first this might seem surprising, but here's why:

1. **Fast-rotating dimensions (3-4)** barely changed between positions:
   - Position 2: 0.0002 radians
   - Position 5: 0.0005 radians
   - These dimensions maintain strong similarity across all these positions
   - They contribute positively to both attention scores

2. **Slow-rotating dimensions (1-2)** show more differentiation:
   - Adjacent tokens: 0.01 radian difference → still well-aligned
   - Distant tokens: 0.03 radian difference → slightly less aligned
   - **This is where the decay begins!**

3. **At much longer distances** (e.g., distance = 50), the decay becomes dramatic:
   - Slow dims: $50 \times 0.01 = 0.5$ radians (≈29°) → significant misalignment
   - Fast dims: $50 \times 0.0001 = 0.005$ radians (≈0.3°) → still mostly aligned
   - Now slow dimensions contribute negatively or weakly

### The Multi-Frequency Insight

This demonstrates **why RoPE uses multiple rotation frequencies**:

- **Slow $\theta_1$** (dimensions 1-2): Sensitive to position differences, even at short range. Captures that "cat" and "mouse" are separated.

- **Fast $\theta_2$** (dimensions 3-4): Maintains similarity across these short distances. Useful for capturing that both tokens are in the same local context (same sentence).

**The model learns during training which frequency to rely on for different linguistic patterns!**

### The Key Property: Only Relative Distance Matters

For "cat" → "mouse", the attention computation uses:

$$R_2^T R_5 = R_{5-2} = R_3$$

This means the rotation is by the **relative distance** of 3 positions:
- Dimensions 1-2: $3 \times 0.01 = 0.03$ radians
- Dimensions 3-4: $3 \times 0.0001 = 0.0003$ radians

**This relative rotation is identical for ANY two tokens 3 positions apart**, regardless of their absolute positions!

**Example:** 
- Tokens at positions (2, 5): relative rotation = $R_3$
- Tokens at positions (10, 13): relative rotation = $R_3$
- Tokens at positions (100, 103): relative rotation = $R_3$

All three pairs get the **same positional relationship** encoded!

### Comparing Attention Across Different Distances

Let's trace how attention changes as distance increases:

**Distance 1 (adjacent tokens):**
- Slow dims: $1 \times 0.01 = 0.01$ rad (0.6°) → barely rotated → **strong contribution**
- Fast dims: $1 \times 0.0001 = 0.0001$ rad (0.006°) → essentially identical → **strong contribution**
- **Result: HIGH total attention** (as we computed: 1.004)

**Distance 3 (our example):**
- Slow dims: $3 \times 0.01 = 0.03$ rad (1.7°) → slight rotation → **moderate contribution**
- Fast dims: $3 \times 0.0001 = 0.0003$ rad (0.02°) → barely changed → **strong contribution**
- **Result: STILL HIGH attention** (as we computed: 1.014)

**Distance 50 (far apart):**
- Slow dims: $50 \times 0.01 = 0.5$ rad (29°) → significant rotation → **weak or negative contribution**
- Fast dims: $50 \times 0.0001 = 0.005$ rad (0.3°) → noticeable but small → **moderate contribution**
- **Result: LOWER total attention**

**Distance 500 (extremely far):**
- Slow dims: $500 \times 0.01 = 5.0$ rad (286°) → nearly opposite direction → **negative contribution!**
- Fast dims: $500 \times 0.0001 = 0.05$ rad (3°) → finally noticeable → **weakening contribution**
- **Result: VERY LOW or negative attention**

### Why This Matters: Long-Term Decay Property

This is the **long-term decay property** in action:

- **Short distances:** Both frequency bands contribute positively → strong attention
- **Medium distances:** Slow frequencies start diverging → moderate attention  
- **Long distances:** Slow frequencies oppose, fast frequencies weaken → low attention
- **Very long distances:** Both frequency bands contribute weakly or negatively → minimal attention

**The decay happens naturally and automatically** across multiple scales, without any explicit distance penalty! The model learns to use different frequency bands for different linguistic relationships during training.
---

## Question 1: Sequence Length Flexibility

**Q:** Traditional position embeddings use a fixed matrix $W_p \in \mathbb{R}^{d_e \times \ell_{\max}}$ that can only handle sequences up to length $\ell_{\max}$. How does RoPE enable models to handle sequences **longer** than those seen during training? Why can a model trained on 512-token sequences successfully process 2048-token sequences at inference?

<details>
<summary>Click to reveal answer</summary>

**Answer:** RoPE encodes **relative distances**, not absolute positions, which makes it inherently flexible to sequence length.

**Why traditional encoding fails:**

With additive position encoding, $W_p$ has exactly $\ell_{\max}$ columns - one for each position:
- Position 1 gets column 1: $W_p[:, 1]$
- Position 2 gets column 2: $W_p[:, 2]$
- ...
- Position 512 gets column 512: $W_p[:, 512]$

**What about position 513?** There is no column 513! The model has never seen how to represent positions beyond $\ell_{\max}$.

**Why RoPE succeeds:**

RoPE doesn't store position embeddings - it **computes rotations** on the fly:

$$R_m = \begin{bmatrix} \cos(m\theta) & -\sin(m\theta) \\ \sin(m\theta) & \cos(m\theta) \end{bmatrix}$$

Key insights:

1. **Rotation is a continuous function**: We can compute $R_m$ for ANY value of $m$, not just $m \leq 512$
   - If trained on $m \in [1, 512]$, we can still compute $R_{1000}$, $R_{2048}$, etc.
   - It's just $\cos(1000\theta)$ and $\sin(1000\theta)$ - basic math!

2. **Attention depends on relative distance**: 
   - Two tokens at positions $(100, 103)$ have relative distance 3
   - Two tokens at positions $(1000, 1003)$ also have relative distance 3
   - Both pairs get the SAME attention relationship: $R_{100}^T R_{103} = R_3 = R_{1000}^T R_{1003}$

3. **Model learns relative patterns, not absolute positions**:
   - During training, the model learns "tokens 3 positions apart should attend like this"
   - At inference with longer sequences, those same relative patterns apply
   - Position 1005 attending to position 1008? That's distance 3 - model already knows how to handle it!

**Concrete example:**

Training: "The cat sat" (positions 1, 2, 3)
- "cat" → "sat": distance 1, rotation $R_1$
- Model learns: "adjacent tokens have strong attention"

Inference: Longer document with "...the dog barked..." at positions 1000, 1001, 1002
- "dog" → "barked": distance 1, rotation $R_{1001}^T R_{1002} = R_1$ (same as training!)
- Model uses the same learned pattern for adjacent tokens

**The mathematical guarantee:**

Since ${R_m}^T R_n = R_{(n-m)}$ always holds, the attention computation for any pair of tokens depends only on $(n-m)$, which the model has seen during training even if the absolute positions $m$ and $n$ are new!

**Contrast with absolute position:**

If position 1005 got a unique embedding $p_{1005}$, the model would need to have learned what $p_{1005}$ means - but it never saw positions beyond 512 during training!

This is why RoPE enables **length extrapolation**: the model generalizes to longer sequences because it learned relative relationships that are valid at any absolute position.

</details>

---

## Why Rotation Works Better Than Addition

### Mathematical Comparison

**Traditional (Addition):**

$$q_m = W_q(x_m + p_m)$$
$$k_n = W_k(x_n + p_n)$$

$$q_m^T k_n = (W_q x_m + W_q p_m)^T (W_k x_n + W_k p_n)$$

Expanding this:

$$= x_m^T W_q^T W_k x_n + x_m^T W_q^T W_k p_n + p_m^T W_q^T W_k x_n + p_m^T W_q^T W_k p_n$$

This has **FOUR terms** with absolute positions $m$ and $n$ appearing separately. The model must learn to extract relative position from these mixed terms.

**RoPE (Rotation):**

$$q_m = R_m W_q x_m$$
$$k_n = R_n W_k x_n$$

$$q_m^T k_n = x_m^T W_q^T R_m^T R_n W_k x_n = x_m^T W_q^T R_{(n-m)} W_k x_n$$

Only **ONE term** appears, and it directly contains the **relative distance $(n-m)$** because:

$${R_{\Theta,m}^d}^T R_{\Theta,n}^d = R_{\Theta,n-m}^d \quad \text{(property of rotation matrices)}$$

---

## Question 2: The Special Property of Rotation Matrices

**Q:** We've seen that rotation has the property ${R_m}^T R_n = R_{(n-m)}$, which directly encodes relative position. But why is this property unique to rotation? What makes rotation matrices special compared to other transformations, and why can't we achieve the same effect with traditional additive position encoding?

<details>
<summary>Click to reveal answer</summary>

**Answer:** The property ${R_m}^T R_n = R_{(n-m)}$ comes from fundamental properties of rotation matrices as elements of a mathematical group. This algebraic structure is what makes rotation special.

### The Group Property of Rotations

Rotation matrices form a **group** under composition, meaning:
1. **Closure:** Rotating by $m$ then by $n$ gives another rotation
2. **Inverse:** $R_m^T = R_{-m}$ (rotating backwards undoes the rotation)
3. **Composition rule:** $R_m \cdot R_n = R_{m+n}$

From property 2 and 3:
$$R_m^T R_n = R_{-m} \cdot R_n = R_{n-m}$$

**This is the key!** The inverse of a rotation by $m$ followed by a rotation by $n$ is equivalent to a rotation by the **difference** $(n-m)$.

### Why This Encodes Relative Position

When computing attention $q_m^T k_n$:

$$q_m^T k_n = (R_m W_q x_m)^T (R_n W_k x_n) = x_m^T W_q^T \underbrace{R_m^T R_n}_{R_{(n-m)}} W_k x_n$$

The rotations "compose" through matrix multiplication, and by the group property, $R_m^T R_n = R_{(n-m)}$.

**Result:** The attention score depends only on the **relative distance** $(n-m)$, not the absolute positions $m$ and $n$ individually!

### Why Addition Doesn't Have This Property

With additive encoding, positions enter as simple vector addition:

$$q_m = W_q(x_m + p_m)$$
$$k_n = W_k(x_n + p_n)$$

When we compute $q_m^T k_n$:

$$q_m^T k_n = (W_q x_m + W_q p_m)^T (W_k x_n + W_k p_n)$$

Expanding:
$$= \underbrace{x_m^T W_q^T W_k x_n}_{\text{content-content}} + \underbrace{x_m^T W_q^T W_k p_n}_{\text{content-position}} + \underbrace{p_m^T W_q^T W_k x_n}_{\text{position-content}} + \underbrace{p_m^T W_q^T W_k p_n}_{\text{position-position}}$$

**Four separate terms!** Each contains absolute positions $m$ and $n$ independently:
- $p_m$ appears in two terms
- $p_n$ appears in two terms
- There's no automatic "cancellation" that leaves only $(n-m)$

The model must **learn** to extract relative position from these mixed terms - it's not automatically encoded like with rotation.

### Mathematical Intuition: Why Group Structure Matters

**Rotation:** A geometric transformation with algebraic structure
- $R_{m+n} = R_m \cdot R_n$ → rotations compose
- $R_{-m} = R_m^{-1} = R_m^T$ → rotations have inverses
- These properties guarantee: $R_m^T R_n = R_{n-m}$ → **relative distance emerges automatically**

**Addition:** A simple arithmetic operation without geometric structure
- $(x + p_m)$ and $(x + p_n)$ are just sums
- No composition rule that cancels absolute positions
- No guarantee that relative distance $(n-m)$ appears naturally

### Visual Analogy

**Rotation (RoPE):**
- Imagine people standing on a clock face at different hour marks
- Person A at 2 o'clock, Person B at 5 o'clock
- The **angular distance** between them is always 3 hours, regardless of where you rotate the clock
- The relationship is **intrinsic** to their positions on the circle

**Addition (Traditional):**
- Imagine painting Person A blue (color = position 2) and Person B yellow (color = position 5)
- To know their distance, you must remember: "blue means 2, yellow means 5, so distance is 3"
- The relationship is **learned**, not intrinsic

### Why This Matters for Linear Attention

Linear attention decomposes attention as:

$$\text{Attention}(Q, K, V) = \phi(Q)[\phi(K)^T V]$$

This requires that $Q$ and $K$ can be processed **independently** - we can't have complex interactions between $q_m$ and $k_n$ before combining them.

**With addition:** The four-term expansion means $q_m$ and $k_n$ interact in complex ways → breaks the linear decomposition

**With rotation:** The clean composition $R_m^T R_n = R_{(n-m)}$ preserves the structure needed for linear attention → $O(N)$ complexity is possible!

### The Fundamental Insight

**Rotation matrices have algebraic properties (group structure) that make relative position encoding "automatic".**

Other transformations (addition, scaling, etc.) don't have these properties, so relative position must be learned from absolute positions - less elegant and less effective.

This is why the paper derives RoPE from first principles: start with the desired property (relative position), then ask "what mathematical operation naturally has this property?" The answer: rotation!

</details>

---

## Impact and Performance Benefits

### Real-World Adoption

RoPE has become the **de facto standard** for position encoding in modern large language models:

**Major Models Using RoPE:**
- **LLaMA** (Meta, 2023): Uses RoPE as default position encoding, achieves SOTA on many benchmarks
- **PaLM** (Google, 2022): Uses RoPE, successfully handles 8K+ token contexts
- **GPT-NeoX** (EleutherAI): Rotary embeddings enable efficient long-context modeling
- **Default in HuggingFace Transformers** for many architectures

### Demonstrated Performance Benefits

**1. Faster Convergence in Pre-training**
- RoFormer converges faster than vanilla BERT during masked language modeling
- Experiments show same performance reached in **fewer training steps**
- Lower compute costs for pre-training

**Evaluated on:**
- Masked Language Modeling (MLM) on Chinese Wikipedia corpus
- Compared training curves against BERT baseline

**2. Superior Performance on Long Text Tasks**

**The authors report:** "The experimental results show that our proposed RoFormer can achieve better performance on long texts task."

- **+1.5% accuracy** on CAIL2019-SCM (Chinese legal document classification) with 1024-token context vs 512
- **+0.2 BLEU** improvement on WMT 2014 En-De machine translation
- Performance continues **improving** as sequence length increases (not true for traditional approaches)

**Tasks evaluated:**
- **Machine Translation:** WMT 2014 English-to-German (seq-to-seq, handles variable-length inputs/outputs)
- **Long Text Classification:** CAIL2019-SCM Chinese legal case judgment prediction (documents 512-1024 tokens)
- **Question Answering:** CMRC 2018 Chinese machine reading comprehension
- **Natural Language Inference:** XNLI cross-lingual inference tasks
- **Sentiment Analysis:** ChnSentiCorp Chinese sentiment classification

**Key finding:** RoPE's advantage increases with sequence length, particularly evident on Chinese long-text tasks where documents exceed 512 tokens.

**3. Length Generalization (Enabled by Relative Encoding)**
- Train on 512-token sequences
- Generalize to 2048+ tokens at inference **without retraining**
- Traditional absolute position embeddings fail at unseen lengths

**4. Linear Attention Compatibility**
- RoPE works with $O(N)$ linear attention mechanisms
- Traditional additive encoding breaks linear attention decomposition
- Critical for scaling to 100K+ token contexts
### Core Architectural Properties

**1. Explicit Relative Position Encoding**
- Attention naturally depends only on token distance, not absolute positions
- Mathematical guarantee: ${R_{\Theta,m}^d}^T R_{\Theta,n}^d = R_{\Theta,n-m}^d$

**2. Long-Term Decay Property**
- Attention scores automatically decrease with distance
- Nearby tokens get more attention than distant ones (linguistically intuitive)
- Aligns with natural language structure

**3. Sequence Length Flexibility**
- No maximum sequence length baked into the model
- Can handle longer contexts at inference than seen during training
- Enables practical deployment on variable-length inputs

**4. Multiplicative Position Encoding**
- Position enters through rotation (multiplicative), not addition
- Keeps content and position information separate
- Cleaner mathematical structure

---

## Impact on the AI Landscape

### Paradigm Shift: From Additive to Multiplicative Encoding

**Before RoPE:** Position encoding meant "add vectors to embeddings"

**After RoPE:** Position can be encoded through **operations** (rotation, bias, etc.)

This opened new research directions:
- **ALiBi** (2022): Adds bias to attention scores instead of embeddings
- **Relative position representations** becoming standard in modern architectures
- Rethinking: what should be "in the data" vs "in the operation"?

### Bridge Between Theory and Practice

RoFormer exemplifies **principle-driven design:**
1. Start with desired mathematical properties (relative position dependency)
2. Derive architecture from first principles
3. Validate empirically

This methodology contrasts with trial-and-error architecture search and influenced subsequent work to be more theory-grounded.

### Intersection with Other Work

**Past work it built on:**
- **Transformer-XL** (2019): Introduced relative position encoding via decomposition
- **Complex-valued networks**: Use of rotation in complex space

**Present impact:**
- **Most widely adopted** relative position encoding in modern LLMs
- Default choice in HuggingFace Transformers implementations

**Future directions enabled:**
- **Extrapolation to longer sequences:** Length flexibility allows generalization
- **Linear attention with position:** $O(N)$ complexity with position encoding
- **Geometric inductive biases:** Opens questions about other geometric transformations

---
## Critical Analysis

### What Was Overlooked or Could Be Developed Further?

**1. Limited Theoretical Analysis of Why It Converges Faster**

The paper shows RoFormer converges faster than BERT empirically (Figure 3), but doesn't provide deep theoretical explanation of WHY. They prove long-term decay (Section 3.4.3) but don't connect this to optimization dynamics.

**The authors explicitly acknowledge this limitation (Section 4.5.5):** "Although we have proved that our model has favourable property of long-term decay for inter-token products, Section (3.3), which is similar to the existing position encoding mechanisms, our model shows superior performance on long texts than peer models, we have not come up with a faithful explanation."

**What they prove:** Long-term decay property exists  
**What remains unexplained:** Why this property leads to faster convergence and better long-text performance

**Potential development:** Analyze the loss landscape properties that RoPE induces. Does the explicit relative position structure create smoother gradients? Does the multiplicative encoding create better conditioning for optimization?

**2. Interaction with Different Attention Patterns**

The paper briefly mentions sparse attention (GPT-3 uses it) but doesn't deeply explore how RoPE interacts with different attention patterns like:
- Local attention windows
- Sliding window attention  
- Block-sparse attention

Given that RoPE naturally creates position-dependent attention scores, understanding its interaction with attention pattern constraints could reveal optimization opportunities.

**3. Extension to Other Modalities**

RoPE is derived for sequential 1D data. What about:
- 2D positional encoding for images (where position is $(x, y)$)?
- 3D for video $(x, y, t)$)?
- Graph structures where "position" is less clear?

**The authors note:** "The experimental results also show that our proposed RoFormer can achieve better performance on long texts task" but don't explore whether this extends to other structured data types.

**Subsequent work** (e.g., RoPE-2D, RoPE-3D) has begun addressing this, showing the core principle generalizes beyond 1D sequences.

**4. Computational Overhead Not Fully Analyzed**

While they provide efficient implementation (Section 3.4.2), they don't benchmark actual wall-clock time vs traditional approaches at scale. Questions remain:
- What is the actual throughput (tokens/second) compared to additive encoding?
- Does the 2× theoretical operation count translate to 2× wall-clock time, or do optimizations close this gap?
- At what sequence length does the overhead become prohibitive?

### Errors or Disputed Findings?

**No major errors found.** The mathematical derivation is sound and the experimental results are reproducible.

**Key limitation acknowledged by authors** (Section 4.5.5): They note they lack "thorough explanations on why it converges faster" and why it performs better on long texts beyond the decay property. This is an important gap between empirical success and theoretical understanding.

**Open question:** The paper claims RoPE has "sequence length no constraint" due to relative position encoding, but practical limits exist:
- Numerical precision at extreme sequence lengths (e.g., 1M+ tokens)
- Does the decay property remain beneficial at all scales?
- Are there sequence lengths where the approximation breaks down?

**Subsequent work** (ALiBi, 2021) has shown even simpler approaches can work well, suggesting RoPE may not be the only solution - but it remains the most widely adopted, likely due to its principled mathematical foundation and strong empirical results.

### What Makes This Work Strong Despite Limitations

**1. Principle-Driven Design**
- Started with desired property (relative position dependency)
- Derived solution from first principles  
- Created reproducible methodology for future work

**2. Empirical Validation Across Tasks**
- Machine translation, language modeling, long-text classification
- Consistent improvements across diverse benchmarks

**3. Practical Impact**
- Widely adopted in production systems (LLaMA, PaLM, GPT-NeoX)
- Enabled real-world long-context applications

The gap between "we know it works" and "we know why it works" is not unique to this paper - it reflects a broader challenge in deep learning research. RoFormer's contribution is providing a solution that is both theoretically motivated AND empirically successful.

---

## Key Takeaways: Two Concepts for Future Development

### 1. Relative > Absolute for Structural Relationships

**Insight:** For many domains, what matters is **how elements relate** to each other, not their absolute properties.

**In language:** "How far apart are these words?" matters more than "what position is this word?"

**Broader implications:**
- **Time series:** Gaps between events matter more than exact timestamps
- **Graphs:** Edge distances matter more than node IDs
- **Images:** Spatial relationships matter more than pixel coordinates
- **Any structured data:** Consider encoding relationships, not just attributes

**Future research question:** What other domains benefit from relative encoding? How do we identify when absolute vs relative position matters?

### 2. Geometric Inductive Biases Through Algebraic Structure

**Insight:** Rotation matrices have **group structure** that perfectly encodes linguistic properties (relative distance, decay). This wasn't obvious - it was derived from first principles!

**The key:** Find mathematical operations whose **algebraic properties** match the domain structure:
- Rotation → relative distance (through ${R_m}^T R_n = R_{(n-m)}$)
- What else?

**Questions for future work:**
- **Scaling** for magnitude relationships? (e.g., "importance" increasing logarithmically)
- **Reflection** for symmetry? (e.g., palindromes, mirror structures)
- **Shearing** for hierarchy? (e.g., tree structures, organizational charts)
- **Other Lie groups** for different structural properties?

**Why this matters:** 
Instead of hoping neural networks learn structure from data, we can **bake in inductive biases** through carefully chosen geometric transformations. This is the essence of principle-driven AI design!

**The methodology:**
1. Identify desired structural properties (e.g., "relative position should matter")
2. Find mathematical operations with matching algebraic structure (e.g., rotation groups)
3. Derive architecture from these operations
4. Validate empirically

This is more elegant and data-efficient than pure empirical architecture search!

---

## Resources

**Original Paper:** https://arxiv.org/abs/2104.09864

**HuggingFace Implementation:** https://huggingface.co/docs/transformers/model_doc/roformer

**Illustrated Blog (EleutherAI):** https://blog.eleuther.ai/rotary-embeddings/

**Official GitHub:** https://github.com/ZhuiyiTechnology/roformer

**Follow-up Work - ALiBi:** https://arxiv.org/abs/2108.12409

**LLaMA Paper (RoPE in practice):** https://arxiv.org/abs/2302.13971

---

## Code Demonstration

See the Jupyter notebook `roformer_demo.ipynb` in this repository for:

1. **Implementing RoPE from scratch** in NumPy (50 lines)
2. **Comparing attention scores** between traditional and RoPE approaches
3. **Visualizing rotation matrices** at different positions
4. **Testing on "the cat chased the mouse"** example with real numbers
5. **Using HuggingFace RoFormer** for language modeling

---

## Appendix: Connection to Formal Algorithms Paper

For those following our course's Formal Algorithms framework:

**RoFormer modifies:**
- **Section 5: Architectural Components** → Specifically the Attention mechanism
- **Algorithm 4: Attention** → Lines 1-3 change from additive to multiplicative
- **Algorithm 2: Positional Embedding** → Eliminated entirely!

**Key algorithmic change:**

**Traditional:**
$$e_t = W_e[:, x[t]] + W_p[:, t] \quad \text{(Algorithm 1 + 2)}$$
$$Q = W_q E + b_q \mathbf{1}^T \quad \text{(Algorithm 4, line 1)}$$

**RoFormer:**
$$e_t = W_e[:, x[t]] \quad \text{(Algorithm 1 only)}$$
$$Q = [R^d_{\Theta,m} W_q e_m : m \in [\ell]] \quad \text{(Modified Algorithm 4)}$$

The rotation matrix $R^d_{\Theta,m}$ is a **hyperparameter** (determined by $\theta_i = 10000^{-2i/d}$), not a learned parameter.

**NotebookLM Podcast:** https://notebooklm.google.com/notebook/ac3b5748-eedb-4555-9008-8a71af96f425

---

**Questions during presentation? Feel free to interrupt!**
