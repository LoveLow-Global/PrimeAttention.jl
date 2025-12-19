# PrimeAttention.jl

![Prime Attention Heatmap](images/prime_attention_heatmap.png)

A sparse, number-theoretic attention mechanism library for [Flux.jl](https://fluxml.ai/Flux.jl/stable/).

## Overview

**PrimeAttention.jl** provides sparse attention layers that leverage number-theoretic sequences to define connectivity patterns. By using mathematical sequences instead of dense matrices or random sparsity, these layers achieve long-range dependencies with significantly lower computational complexity than standard $O(N^2)$ Transformers.

Inspired by architectures like [BigBird](https://arxiv.org/abs/2007.14062), this package implements a hybrid mechanism that combines three strategies:
1.  **Global Tokens:** For sequence-wide context summaries.
2.  **Sliding Window:** For local syntax and immediate context.
3. **Theoretic Intervals:** Sparse long-range connections based on Primes, Squares, or the Mian-Chowla sequence.

## Usage

The package provides factory functions that return a `SparseIndexAttention` layer. Here is a simple example.

```julia
using Flux
using PrimeAttention


# Define parameters
embed_dim = 64
n_heads = 4
global_tokens = 2
window_size = 3

# Initialize layer
attention_layer = PrimeSelfAttention(embed_dim; heads=n_heads, global_tokens=global_tokens, window=window_size)

# Dummy input
x = rand(Float32, embed_dim, 128, 8)

# Forward pass
y = attention_layer(x)

println("Input:  ", size(x))
println("Output: ", size(y))

# Inside a Flux Chain
model = Chain(
    Dense(32 => 64),
    PrimeSelfAttention(64; heads=4, global_tokens=2, window=3), 
    LayerNorm(64),
    Dense(64 => 10),
    softmax
)
```

## Layer

While not linear, this offers a significant speedup over standard $O(N^2)$ attention.

| Layer | Sequence | Density ($i$-th token) | Complexity | Best For |
| :--- | :--- | :--- | :--- | :--- |
| `PrimeSelfAttention` | Primes $\{2, 3, 5, \dots\}$ | $\approx 1/\ln i$ | $O(N^2 / \ln N)$ | Balanced long-range skip |
| `SquareSelfAttention` | Squares $\{1, 4, 9, \dots\}$ | $\approx 1/\sqrt{i}$ | $O(N \sqrt{N})$ | Extremely sparse / Efficiency |
| `MianChowlaSelfAttention` | Mian-Chowla $\{1, 2, 5, \dots\}$ | $< 1/\sqrt{i}$ | $O(N \sqrt{N})$ | Non-redundant "Sidon" patterns | 


## Number-Theoretic Background

According to the Prime Number Theorem, the density of prime numbers decreases as numbers get larger, approximately following $\pi(x) \approx x/\ln x$.

By restricting attention connections to relative distances $p \in \{2, 3, 5, 7, 11, \dots\}$, PrimeAttention achieves a natural **"Fading Attention"** mechanism:
* **High Resolution:** The model retains dense connectivity in the recent past, as small primes are frequent.
* **High Efficiency:** The model utilizes sparse connectivity in the distant past, as large primes are rare.
* **Deterministic Irregularity:** Unlike fixed stride patterns, prime intervals avoid harmonic synchronization, reducing blind spots in the receptive field.

## Architecture

For a query token at index $i$ and a key token at index $j$, the causal mask is defined as

$$A_{i,j} = 
\begin{cases} 
1 & \text{if } j \leq G & \text{(Global)} \\
1 & \text{if } i - j \leq W & \text{(Window)} \\
1 & \text{if } (i - j) \in \mathcal{S} & \text{(Sequence } \mathcal{S}\text{)} \\
0 & \text{otherwise}
\end{cases}$$

Where $G$ is the number of global tokens and $W$ is the window size.


## Implementation Notes

Refactored Kernel: All layers share a universal `sparse_index_kernel` that iterates exclusively over valid indices.

Differentiation: Fully compatible with `Zygote.jl` via `Zygote.Buffer` to handle array mutations during the forward pass.

Performance: While complexity is reduced mathematically, speedup in the current version is limited by CPU-based iteration. This implementation serves as a research reference mainly.