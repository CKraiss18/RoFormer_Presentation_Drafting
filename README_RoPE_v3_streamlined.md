# RoFormer: Enhanced Transformer with Rotary Position Embedding

**Presenter:** Charlee Kraiss  
**Course:** DS 5690-01 Gen AI Models in Theory and Practice (2025F)  
**Date:** 30 - Oct - 2025

---

## Paper Information

**Title:** RoFormer: Enhanced Transformer with Rotary Position Embedding

**Authors:** Jianlin Su, Yu Lu, Shengfeng Pan, Ahmed Murtadha, Bo Wen, Yunfeng Liu (Zhuiyi Technology Co., Ltd.)

**Published:** November 9, 2023 (arXiv:2104.09864v5)

**Paper Link:** https://arxiv.org/abs/2104.09864

---

## Why RoPE Matters: Impact and Performance Benefits

### Real-World Adoption

RoPE has become the **de facto standard** for position encoding in modern large language models:

- **LLaMA** (Meta, 2023): Uses RoPE as default, achieves SOTA on many benchmarks
- **PaLM** (Google, 2022): Uses RoPE, handles 8K+ token contexts
- **GPT-NeoX** (EleutherAI): Rotary embeddings enabled efficient long-context modeling

**Why did these models choose RoPE over traditional approaches?**

### Demonstrated Performance Benefits

**1. Faster Convergence in Pre-training**
- RoFormer converges faster than vanilla BERT during masked language modeling
- Same performance reached in fewer training steps → lower compute costs

**2. Superior Performance on Long Text Tasks**
- **+1.5% accuracy** on CAIL2019-SCM (Chinese legal documents) with 1024-token context vs 512
- Performance continues improving as sequence length increases (not true for traditional approaches)
- **+0.2 BLEU** on WMT 2014 En-De machine translation

**3. Length Generalization**
- Train on 512 tokens, generalize to 2048+ tokens **without retraining**
- Traditional absolute position embeddings fail at unseen lengths

### Core Architectural Properties Enabled by RoPE

**1. Explicit Relative Position Encoding**
- Attention naturally depends only on token distance, not absolute positions
- Mathematical guarantee: ${R_{\Theta,m}^d}^T R_{\Theta,n}^d = R_{\Theta,n-m}^d$

**2. Long-Term Decay Property**
- Attention scores automatically decrease with distance
- Nearby tokens get more attention than distant ones (linguistically intuitive)

**3. Sequence Length Flexibility**
- No maximum sequence length baked into the model
- Enables handling longer contexts at inference than seen during training

**4. Linear Attention Compatibility**
- Works with $O(N)$ attention mechanisms (traditional additive encoding breaks this)
- Critical for scaling to 100K+ token contexts

---

## The Core Problem

### Transformers Are Position-Agnostic

Without positional information, transformers treat sequences as "bags of words."

**Example:** "the cat chased the mouse" vs "the mouse chased the cat"
- Same tokens → identical representations
- Meaning completely changes, but model sees no difference

### Previous Solution: Additive Position Encoding

From our Formal Algorithms paper (Algorithm 2):

$$e_t = W_e[:, x[t]] + W_p[:, t]$$

**Add** position embedding $W_p[:, t]$ to token embedding $W_e[:, x[t]]$

**Problems:**
1. Position and content information get mixed in embedding space
2. Relative distances aren't explicitly encoded in attention
3. Maximum sequence length is fixed by $W_p$ dimensions
4. Doesn't work with linear attention ($O(N)$ complexity)

---

## RoPE's Solution: Rotate, Don't Add

### The Key Insight

Instead of **adding** position information to embeddings, **rotate** query and key vectors during attention computation.

**Mathematical property of rotation:**
$$\text{Rotation by angle } m\theta \text{ then by } n\theta = \text{Rotation by } (n-m)\theta$$

This naturally encodes **relative distance** $(n-m)$, not absolute positions $(m, n)$!

### How It Works (High Level)

**Traditional Attention:**
```
1. Token Embedding: x
2. Add Position: x + p_m
3. Compute Q, K: W_q(x + p_m), W_k(x + p_n)
4. Attention: (x + p_m)^T (x + p_n)
```

**RoPE Attention:**
```
1. Token Embedding: x
2. Compute Q, K: W_q x, W_k x
3. Rotate by Position: R_m(W_q x), R_n(W_k x)
4. Attention: [R_m(W_q x)]^T [R_n(W_k x)] = ... R_(n-m) ...
```

Position enters **during** attention computation, not **before** it.

---

## Algorithm: RoPE Attention

Modified from Formal Algorithms Algorithm 4:

**Input:** 
- $X \in \mathbb{R}^{d_x \times \ell_x}$, $Z \in \mathbb{R}^{d_z \times \ell_z}$, token sequences

**Output:** 
- $\tilde{X} \in \mathbb{R}^{d_{\text{out}} \times \ell_x}$, updated representations

**Parameters:** 
- $W_q \in \mathbb{R}^{d_{\text{attn}} \times d_x}$ (no bias for Q, K with RoPE!)
- $W_k \in \mathbb{R}^{d_{\text{attn}} \times d_z}$
- $W_v \in \mathbb{R}^{d_{\text{out}} \times d_z}$, $b_v \in \mathbb{R}^{d_{\text{out}}}$

**Hyperparameters:** 
- $\Theta = \{\theta_i = 10000^{-2(i-1)/d} : i \in [d/2]\}$

**Algorithm:**
1. **for** $m \in [\ell_x]$ **do** $\triangleright$ Construct rotation matrix for each position

The rotation matrix $R^d_{\Theta,m}$ is block-diagonal:

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

Each 2×2 block rotates a pair of dimensions:

$$
R(m\theta_i) = 
\begin{bmatrix}
\cos(m\theta_i) & -\sin(m\theta_i) \\
\sin(m\theta_i) & \cos(m\theta_i)
\end{bmatrix}
$$

2. **end for**
3. $Q \leftarrow [R^d_{\Theta,1} W_q X[:,1], R^d_{\Theta,2} W_q X[:,2], \ldots, R^d_{\Theta,\ell_x} W_q X[:,\ell_x]]$ $\triangleright$ Rotate queries
4. $K \leftarrow [R^d_{\Theta,1} W_k Z[:,1], R^d_{\Theta,2} W_k Z[:,2], \ldots, R^d_{\Theta,\ell_z} W_k Z[:,\ell_z]]$ $\triangleright$ Rotate keys
5. $V \leftarrow W_v Z + b_v \mathbf{1}^T$ $\triangleright$ Values NOT rotated
6. $S \leftarrow Q^T K$ $\triangleright$ Compute attention scores
7. Apply masking to $S$ if needed
8. **return** $\tilde{X} \leftarrow V \cdot \text{softmax}(S / \sqrt{d_{\text{attn}}})$

**Key differences from traditional attention:**
- **No position embedding step** (Algorithm 2 eliminated!)
- **Rotation applied to Q and K** instead of adding bias
- **No bias terms for Q, K:** Rotating bias makes it position-dependent, breaking the relative position property
- **Values (V) not rotated:** Only Q and K need position info for attention computation

---

## Understanding Multiple Rotation Frequencies

### Why $\theta_i$ Has a Subscript

For higher dimensions (e.g., $d = 512$), RoPE uses **different rotation speeds** for different dimension pairs:

$$\theta_i = 10000^{-2i/d} \quad \text{for } i = 1, 2, \ldots, d/2$$

This creates a geometric series:
- $\theta_1 = 10000^{-2/d}$ → **slow rotation** (captures long-range dependencies)
- $\theta_2 = 10000^{-4/d}$ → **medium rotation**
- $\theta_{d/2} = 10000^{-1} = 0.0001$ → **fast rotation** (captures local dependencies)

### Intuition: Multi-Scale Position Information

| Frequency | Rotation Speed | What It Captures |
|-----------|---------------|------------------|
| Low ($\theta_1$) | Slow (many positions per full rotation) | **Long-range**: "this word is 50+ tokens away" |
| High ($\theta_{d/2}$) | Fast (rotates significantly per position) | **Local**: "these words are adjacent" |

**Analogy:** Like having multiple clock hands moving at different speeds
- Hour hand (slow) → tells you the general time of day
- Minute hand (fast) → tells you the exact minute

---

## Concrete Example: "The Cat Chased The Mouse"

### Setup

Simplified to show the mechanics clearly:
- **Sentence:** "the cat chased the mouse"
- **Positions:** 1, 2, 3, 4, 5
- **Dimensions:** $d = 2$ (instead of 768)
- **Rotation angle:** $\theta = 1.0$ radian (simplified from $\theta_1 = 0.0001$)

### Step 1: Token Embeddings (No Position Yet)

$$
X = 
\begin{bmatrix}
0.5 & 0.9 & 0.2 & 0.5 & 0.3 \\
0.3 & 0.4 & 0.8 & 0.3 & 0.7
\end{bmatrix}
$$

Columns: ["the", "cat", "chased", "the", "mouse"]

### Step 2: Compute Queries and Keys

Using $W_q = W_k = I$ (identity) for simplicity:

$$Q = W_q X = X$$
$$K = W_k X = X$$

### Step 3: Apply Rotations

**For "cat" at position $m = 2$:**
- Rotation angle: $2 \times 1.0 = 2.0$ radians

$$
R_2 = \begin{bmatrix} \cos(2.0) & -\sin(2.0) \\ \sin(2.0) & \cos(2.0) \end{bmatrix} = \begin{bmatrix} -0.416 & -0.909 \\ 0.909 & -0.416 \end{bmatrix}
$$

$$
q_2 = R_2 \begin{bmatrix} 0.9 \\ 0.4 \end{bmatrix} = \begin{bmatrix} -0.738 \\ 0.652 \end{bmatrix}
$$

**For "mouse" at position $n = 5$:**
- Rotation angle: $5 \times 1.0 = 5.0$ radians

$$
R_5 = \begin{bmatrix} \cos(5.0) & -\sin(5.0) \\ \sin(5.0) & \cos(5.0) \end{bmatrix} = \begin{bmatrix} 0.284 & 0.959 \\ -0.959 & 0.284 \end{bmatrix}
$$

$$
k_5 = R_5 \begin{bmatrix} 0.3 \\ 0.7 \end{bmatrix} = \begin{bmatrix} 0.756 \\ -0.089 \end{bmatrix}
$$

### Step 4: Compute Attention Score

$$q_2^T k_5 = (-0.738)(0.756) + (0.652)(-0.089) = -0.616$$

### The Key Property: Relative Position

The attention score **only depends on**:
1. Token embeddings (cat, mouse)
2. **Relative distance:** $5 - 2 = 3$

**NOT** on absolute positions 2 and 5!

**Why?** Because:
$$R_2^T R_5 = R_{5-2} = R_3$$

The rotations "cancel out" to leave only the relative rotation by 3 positions.

---

## Why Rotation Works Better Than Addition

### Mathematical Comparison

**Traditional (Addition):**

$$q_m = W_q(x_m + p_m)$$
$$k_n = W_k(x_n + p_n)$$

Expanding the attention:
$$q_m^T k_n = x_m^T W_q^T W_k x_n + x_m^T W_q^T W_k p_n + p_m^T W_q^T W_k x_n + p_m^T W_q^T W_k p_n$$

**Four terms** with $m$ and $n$ appearing separately! Model must learn to extract relative position from this mess.

**RoPE (Rotation):**

$$q_m = R_m W_q x_m$$
$$k_n = R_n W_k x_n$$

$$q_m^T k_n = x_m^T W_q^T R_m^T R_n W_k x_n = x_m^T W_q^T R_{(n-m)} W_k x_n$$

**One term** with explicit relative distance $(n-m)$ because:

$${R_{\Theta,m}^d}^T R_{\Theta,n}^d = R_{\Theta,n-m}^d \quad \text{(property of rotation matrices)}$$

### Analogy

- **Addition:** Painting people different colors to mark positions. Model must remember "what does red vs blue mean?"
- **Rotation:** Placing people on a clock face. Their angular distance **directly shows** how far apart they are.

---

## Critical Analysis

### What the Paper Overlooked

**1. Limited Theoretical Justification**
- Shows RoPE converges faster empirically, but doesn't explain WHY
- Proves long-term decay property, but doesn't connect to optimization dynamics

**2. Computational Overhead Not Benchmarked**
- Provides efficient implementation, but no wall-clock time comparisons
- How does matrix multiplication for rotation compare to addition?

**3. Extension to Other Modalities**
- RoPE designed for 1D sequences
- What about 2D images, 3D video, graphs?
- Subsequent work (e.g., RoPE-2D) addresses this

### What Makes This Work Strong

**1. Principle-Driven Design**
- Started with desired property (relative position dependency)
- Derived solution from first principles
- Validated experimentally

**2. Clean Mathematical Formulation**
- One simple property: ${R_m}^T R_n = R_{(n-m)}$
- Everything else follows

**3. Practical Impact**
- Widely adopted in production systems
- Enabled models to handle longer contexts efficiently

---

## Impact on the AI Landscape

### Paradigm Shift

**Before RoPE:** Position encoding meant "add vectors to embeddings"

**After RoPE:** Position can be encoded through **operations** (rotation, bias, etc.)

Opened research directions:
- **ALiBi** (2022): Adds bias to attention scores
- **Relative position representations** becoming standard
- Exploration of other geometric transformations

### Why It Changed Practice

**1. Long-Context Requirements**
- Modern applications need 10K-100K+ token contexts
- Traditional approaches hit hard limits
- RoPE enables length extrapolation

**2. Efficiency**
- Compatible with linear attention ($O(N)$ vs $O(N^2)$)
- Critical for scaling to long contexts

**3. Simplicity**
- Cleaner than relative position decompositions (Transformer-XL)
- Easier to implement and understand

---

## Key Takeaways (2 Concepts for Future Development)

### 1. Relative Position > Absolute Position

**Insight:** For language, what matters is "how far apart are these words?" not "where exactly are they?"

**Implication:** Future architectures should encode relational structure, not absolute properties

**Application:** Think about other domains where relative relationships matter:
- Time series: gaps between events matter more than timestamps
- Graphs: edge distances matter more than node IDs
- Images: spatial relationships matter more than pixel coordinates

### 2. Geometric Inductive Biases

**Insight:** Rotation matrices have algebraic properties that perfectly encode linguistic structure

**Question:** What other geometric transformations encode useful biases?
- Scaling for magnitude relationships?
- Reflection for symmetry?
- Shearing for hierarchy?

**Implication:** Don't just throw data at neural networks — design operations that bake in domain knowledge

---

## Resources

- **Paper:** https://arxiv.org/abs/2104.09864
- **HuggingFace Docs:** https://huggingface.co/docs/transformers/model_doc/roformer
- **Code Demo:** See `roformer_demo.ipynb` for implementation and examples
- **EleutherAI Blog:** https://blog.eleuther.ai/rotary-embeddings/

---

## Appendix: Connection to Formal Algorithms

**RoFormer modifies:**
- **Algorithm 4 (Attention):** Lines 1-3 change from additive to rotational
- **Algorithm 2 (Position Embedding):** Eliminated entirely!

**Traditional:**
$$e_t = W_e[:, x[t]] + W_p[:, t]$$
$$Q = W_q E + b_q \mathbf{1}^T$$

**RoFormer:**
$$e_t = W_e[:, x[t]]$$
$$Q = [R^d_{\Theta,m} W_q e_m : m \in [\ell]]$$

The rotation matrix $R^d_{\Theta,m}$ is a **hyperparameter**, not learned.

---

**Questions? Let's discuss!**
