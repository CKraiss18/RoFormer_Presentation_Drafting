# RoFormer: Enhanced Transformer with Rotary Position Embedding

**Presenters:** [Your Name]  
**Course:** DS 5690-01 Gen AI Models in Theory and Practice (2025F)  
**Date:** [Presentation Date]

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

## Overview (5 minutes)

### The Problem

Transformers are **position-agnostic** - without positional information, they treat sequences as "bags of words." In "the cat chased the mouse" vs "the mouse chased the cat," the model would see the same tokens without understanding order.

**Previous solutions** (like in Algorithms 1-2 of our Formal Algorithms paper):
- **Additive position encoding:** Add position vectors to token embeddings
- **Problem:** Position and content get mixed together in embedding space
- **Problem:** Relative distances aren't explicitly encoded in the attention mechanism

### The Approach: Rotary Position Embedding (RoPE)

Instead of **adding** position information to embeddings, RoFormer **rotates** query and key vectors during attention computation.

**Key insight:** Rotation by angle $m$ then rotation by angle $n$ leaves you rotated by angle $(n - m)$, which is the **relative distance**.

### How the Problem Was Addressed

1. **Formulated the goal mathematically:** Position information should only affect attention through relative distances, not absolute positions

2. **Derived the solution:** Starting from first principles (Section 3.1-3.2 of paper), showed that rotation matrices naturally satisfy this property

3. **Proved valuable properties:** Long-term decay, sequence length flexibility, compatibility with linear attention

4. **Validated experimentally:** Improvements on translation, language modeling, and long-text classification

---

## Architecture Overview: Formal Pseudocode

### How Traditional Transformers Encode Position

In our Formal Algorithms paper (Algorithms 1-2), position encoding happens **before** attention:

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

### RoFormer's Approach: Rotation During Attention

RoFormer **eliminates** separate position embedding and instead modifies the attention mechanism:

**Algorithm: RoPE Attention (Modified from Formal Algorithms Algorithm 4)**

**Input:** 
- $X \in \mathbb{R}^{d_x \times \ell_x}$, $Z \in \mathbb{R}^{d_z \times \ell_z}$, token sequences

**Output:** 
- $\tilde{X} \in \mathbb{R}^{d_{\text{out}} \times \ell_x}$, updated representations

**Parameters:** 
- $W_q \in \mathbb{R}^{d_{\text{attn}} \times d_x}$, $b_q \in \mathbb{R}^{d_{\text{attn}}}$
- $W_k \in \mathbb{R}^{d_{\text{attn}} \times d_z}$, $b_k \in \mathbb{R}^{d_{\text{attn}}}$
- $W_v \in \mathbb{R}^{d_{\text{out}} \times d_z}$, $b_v \in \mathbb{R}^{d_{\text{out}}}$

**Hyperparameters:** 
- $\Theta = \{\theta_i = 10000^{-2(i-1)/d} : i \in [d/2]\}$

**Algorithm:**
1. **for** $m \in [\ell_x]$ **do** $\triangleright$ Construct rotation matrix for each position $m$
2. $\quad R^d_{\Theta,m} \leftarrow \text{BlockDiagonal}\left(\begin{bmatrix} \cos(m\theta_1) & -\sin(m\theta_1) \\ \sin(m\theta_1) & \cos(m\theta_1) \end{bmatrix}, \begin{bmatrix} \cos(m\theta_2) & -\sin(m\theta_2) \\ \sin(m\theta_2) & \cos(m\theta_2) \end{bmatrix}, \ldots \right)$
3. **end for**
4. $Q \leftarrow [R^d_{\Theta,1} W_q X[:,1], R^d_{\Theta,2} W_q X[:,2], \ldots, R^d_{\Theta,\ell_x} W_q X[:,\ell_x]]$ $\triangleright$ Apply rotation to queries
5. $K \leftarrow [R^d_{\Theta,1} W_k Z[:,1], R^d_{\Theta,2} W_k Z[:,2], \ldots, R^d_{\Theta,\ell_z} W_k Z[:,\ell_z]]$ $\triangleright$ Apply rotation to keys
6. $V \leftarrow W_v Z + b_v \mathbf{1}^T$ $\triangleright$ Values NOT rotated
7. $S \leftarrow Q^T K$ $\triangleright$ Compute attention scores
8. Apply masking to $S$ if needed
9. **return** $\tilde{X} \leftarrow V \cdot \text{softmax}(S / \sqrt{d_{\text{attn}}})$

**Key differences from Formal Algorithms Algorithm 4:**
- **Line 4-5:** Instead of $Q \leftarrow W_q X + b_q \mathbf{1}^T$, we rotate: $Q \leftarrow [R^d_{\Theta,m} W_q X[:,m]]$
- No separate position embedding step (no Algorithm 2 equivalent)
- Position information appears during attention computation, not in embeddings

---

## A Simple Concrete Example

Let's walk through "the cat chased the mouse" with actual numbers.

### Setup
- **Sentence:** "the cat chased the mouse"
- **Positions:** 1, 2, 3, 4, 5
- **Simplified:** $d = 2$ dimensions, $\theta = 1.0$ radian

### Traditional Transformer

**Token "cat" at position 2:**
```
Token embedding:     W_e[:, cat_id] = [3.0, 4.0]
Position embedding:  W_p[:, 2] = [0.5, 0.3]
Final embedding:     e = [3.0, 4.0] + [0.5, 0.3] = [3.5, 4.3]
```
Position and content are **mixed together** in $[3.5, 4.3]$.

### RoFormer

**Token "cat" at position 2:**

Token embedding only: $W_e[:, \text{cat\_id}] = [3.0, 4.0]$

Position as rotation: Angle $= 2 \times 1.0 = 2.0$ radians

Rotation matrix $R_2$:

$$R_2 = \begin{bmatrix} \cos(2.0) & -\sin(2.0) \\ \sin(2.0) & \cos(2.0) \end{bmatrix} = \begin{bmatrix} -0.416 & -0.909 \\ 0.909 & -0.416 \end{bmatrix}$$

Rotated query:

$$q_2 = R_2 \times \begin{bmatrix} 3.0 \\ 4.0 \end{bmatrix} = \begin{bmatrix} -4.884 \\ 1.063 \end{bmatrix}$$

**Computing attention from "cat" (pos 2) to "mouse" (pos 5):**

- $q_2$ rotated by 2.0 radians: $[-4.884, 1.063]$
- $k_5$ rotated by 5.0 radians: [some vector]
- Attention score $= q_2^T k_5$

The beautiful part: Due to rotation properties, this is mathematically equivalent to:

$$\text{Attention score} = (\text{original vectors}) \text{ rotated by } (5 - 2) = 3 \text{ radians}$$

**The relative distance (3 positions) automatically emerges from the math!**

### Why This Matters

**Distance 1 (nearby tokens):**
- "cat" → "chased": Rotation difference = $1.0$ radian
- Small angle → vectors still relatively aligned → **large attention score**

**Distance 3 (distant tokens):**
- "cat" → "mouse": Rotation difference = $3.0$ radians  
- Large angle → vectors rotated far apart → **smaller attention score**

This is the **long-term decay property** - distant words naturally get less attention!

---

## Questions for the Class

### Question 1: Understanding Relative Position Encoding

**Q:** In the sentence "the cat chased the mouse," using RoFormer, would the positional relationship between "cat" (position 2) and "chased" (position 3) be the **same** as the relationship between "the" (position 4) and "mouse" (position 5)? Why or why not?

<details>
<summary>Click to reveal answer</summary>

**Answer:** YES, they would have the same positional relationship!

**Why:** Both pairs have a relative distance of 1:
- "cat" to "chased": position $3 - 2 = 1$
- "the" to "mouse": position $5 - 4 = 1$

In RoFormer, the attention computation depends on:

$$R^d_{\Theta,m}^T R^d_{\Theta,n} = R^d_{\Theta,n-m}$$

For both pairs, $(n - m) = 1$, so they get the same rotation difference, meaning:
- The geometric relationship between their query and key vectors is identical
- The model treats "1 position apart" consistently regardless of absolute position

This is a **feature, not a bug** - it's the relative position inductive bias that makes RoFormer work well!

</details>

---

### Question 2: Comparing to Traditional Approaches

**Q:** Why can't traditional additive position encoding (adding $W_p[:, t]$ to embeddings) achieve the same relative position encoding as RoFormer? Hint: Think about the mathematical properties of addition vs rotation.

<details>
<summary>Click to reveal answer</summary>

**Answer:** Addition doesn't have the "difference" property that rotation has.

**Mathematical explanation:**

With **addition** (traditional):

$$q_m = W_q(x_m + p_m)$$
$$k_n = W_k(x_n + p_n)$$

$$q_m^T k_n = (W_q x_m + W_q p_m)^T (W_k x_n + W_k p_n)$$

$$= x_m^T W_q^T W_k x_n + x_m^T W_q^T W_k p_n + p_m^T W_q^T W_k x_n + p_m^T W_q^T W_k p_n$$

This has **FOUR terms** with absolute positions $m$ and $n$ appearing separately. The model must learn to extract relative position from these mixed terms.

With **rotation** (RoFormer):

$$q_m = R_m W_q x_m$$
$$k_n = R_n W_k x_n$$

$$q_m^T k_n = x_m^T W_q^T R_m^T R_n W_k x_n = x_m^T W_q^T R_{(n-m)} W_k x_n$$

Only **ONE term** appears, and it directly contains the **relative distance $(n-m)$** because:

$$R_m^T R_n = R_{(n-m)} \quad \text{(property of rotation matrices)}$$

**The key insight:** Rotation has an algebraic property $(R_m^T R_n = R_{(n-m)})$ that directly encodes relative position. Addition has no such property!

**Analogy:** 
- Addition is like painting people different colors to mark their positions - you have to remember "what does red vs blue mean?"
- Rotation is like placing people on a clock face - their angular distance directly shows how far apart they are!

</details>

---

## Critical Analysis

### What Was Overlooked or Could Be Developed Further?

**1. Limited Theoretical Analysis of Why It Converges Faster**

The paper shows RoFormer converges faster than BERT empirically (Figure 3), but doesn't provide deep theoretical explanation of WHY. They prove long-term decay (Section 3.4.3) but don't connect this to optimization dynamics.

**Potential development:** Analyze the loss landscape properties that RoPE induces. Does the explicit relative position structure create smoother gradients?

**2. Interaction with Different Attention Patterns**

The paper briefly mentions sparse attention (GPT-3 uses it) but doesn't deeply explore how RoPE interacts with different attention patterns like:
- Local attention windows
- Sliding window attention  
- Block-sparse attention

**3. Extension to Other Modalities**

RoPE is derived for sequential 1D data. What about:
- 2D positional encoding for images (where position is $(x, y)$)?
- 3D for video $(x, y, t)$?
- Graph structures where "position" is less clear?

**4. Computational Overhead Not Fully Analyzed**

While they provide efficient implementation (Section 3.4.2), they don't benchmark actual wall-clock time vs traditional approaches at scale.

### Errors or Disputed Findings?

**No major errors found.** The mathematical derivation is sound. However:

**Limitation acknowledged by authors (Section 4.5.5):** They note they lack "thorough explanations on why it converges faster" and why it performs better on long texts beyond the decay property.

**Subsequent work** (ALiBi, 2021) has shown even simpler approaches can work well, suggesting RoPE may not be the only solution.

---

## Impact and Significance

### How Did This Work Change the AI Landscape?

**1. Paradigm Shift: From Additive to Multiplicative Position Encoding**

RoFormer demonstrated that position encoding doesn't have to be additive. This opened the door to:
- **ALiBi** (2022): Adds bias to attention scores instead of embeddings
- **Relative position representations** becoming standard in modern architectures
- Rethinking what should be "in the data" vs "in the operation"

**2. Enabled Better Long-Context Models**

The long-term decay and flexible sequence length properties made RoPE attractive for models needing long context:
- **PaLM** (Google, 2022): Uses RoPE, handles 8K+ tokens
- **LLaMA** (Meta, 2023): Uses RoPE as default position encoding
- **GPT-NeoX** (EleutherAI): Uses rotary embeddings

**3. Bridge Between Theory and Practice**

RoFormer exemplifies **principle-driven design:**
- Start with desired mathematical properties (Equation 11: relative position dependency)
- Derive architecture from first principles
- Validate empirically

This methodology contrasts with trial-and-error architecture search and influenced subsequent work to be more theory-grounded.

### Intersection with Other Work

**Past work it built on:**
- Transformer-XL (2019): Introduced relative position encoding via decomposition
- Complex-valued networks: The use of rotation in complex space has roots in prior work on complex-valued neural networks

**Present impact:**
- **Most widely adopted** relative position encoding in modern LLMs
- Default choice in HuggingFace Transformers implementations for many architectures

**Future directions it enables:**
- **Extrapolation to longer sequences:** RoPE's flexibility allows models to generalize to sequence lengths not seen during training
- **Linear attention with position:** Section 3.3 shows RoPE works with $O(N)$ attention, important for scaling
- **Geometric inductive biases:** Opens questions about other geometric transformations (scaling, shearing) for encoding different kinds of structure

### Quantitative Impact

From the paper's experiments:
- **Machine Translation:** +0.2 BLEU improvement on WMT 2014 En-De
- **Long Text Classification:** +1.5% accuracy on CAIL2019-SCM with 1024-token context (vs 512)
- **Pre-training:** Faster convergence (lower loss at same step count vs BERT)

**Real-world adoption (post-paper):**
- LLaMA (65B parameters): Uses RoPE, achieves SOTA on many benchmarks
- Used in production systems requiring long-context understanding

---

## Resource Links

1. **Original Paper:** https://arxiv.org/abs/2104.09864

2. **HuggingFace Implementation:** https://huggingface.co/docs/transformers/model_doc/roformer

3. **Illustrated Blog Post (Jay Alammar style explanation):** https://blog.eleuther.ai/rotary-embeddings/

4. **GitHub Repository (Official):** https://github.com/ZhuiyiTechnology/roformer

5. **Follow-up Work - ALiBi (Alternative approach):** https://arxiv.org/abs/2108.12409

6. **LLaMA Paper (shows RoPE in practice):** https://arxiv.org/abs/2302.13971

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

---

**Questions during presentation? Feel free to interrupt!**
