# End-to-End-Transformer-Systems-Classification-Language-Modeling-and-ALiBi-Optimization

This repository contains a full implementation of Transformer-based models built **from scratch in PyTorch**, including:

- Transformer Encoder for text classification  
- Transformer Decoder for autoregressive language modeling  
- ALiBi positional bias exploration  
- Attention visualization & sanity checks  

This project demonstrates deep understanding of:

- Multi-head self-attention
- Causal masking
- Layer normalization
- Positional encoding (absolute & ALiBi)
- Cross-entropy & perplexity evaluation
- End-to-end training
- Model generalization analysis

---

# Project Structure

| Part | Description |
|------|------------|
| **Part One** | Transformer Encoder + FFN classifier (speech classification) |
| **Part Two** | Transformer Decoder (causal LM pretraining + perplexity) |
| **Part Three** | Architecture exploration with **ALiBi** positional bias |

---

# Architecture Overview

### Encoder (Classification)

```
Token Embedding
    ↓
Positional Encoding
    ↓
Multi-Head Self-Attention × 4
    ↓
LayerNorm
    ↓
Mean Pooling
    ↓
Feedforward Classifier
```

---

### Decoder (Language Modeling)

```
Token Embedding
    ↓
Causal Self-Attention + Mask
    ↓
Transformer Blocks × 4
    ↓
LayerNorm
    ↓
Linear LM Head
```

---

### ALiBi Variant

Instead of absolute positional embeddings:

\[
\text{Attention Score} =
\frac{QK^\top}{\sqrt{d}}
-
m_h \cdot (i - j)
\]

- Injects relative positional bias directly into attention scores
- Improves length generalization
- Adds **zero additional parameters**

---

# Experimental Results

## Decoder Language Modeling (500 iterations)

| Dataset | Perplexity |
|----------|------------|
| Train | ~120 |
| Obama | ~347 |
| W. Bush | ~452 |
| G. H. Bush | ~399 |

ALiBi produced slightly improved perplexities across all test sets, demonstrating improved inductive bias for locality.

---

# Attention Sanity Check

We verify:

- Attention rows sum to 1
- Proper causal masking
- Visualization of attention heatmaps

Generated plots:
```
attention_map_1.png
attention_map_2.png
...
```

---

# How to Run

## Part One — Classification

```bash
python3 main.py --part one
```

Trains encoder + classifier and reports accuracy.

---

## Part Two — Decoder Language Modeling

```bash
python3 main.py --part two
```

Trains causal decoder and reports perplexity on:

- train_LM
- test_LM_obama
- test_LM_wbush
- test_LM_ghbush

---

## Part Three — ALiBi Exploration

```bash
python3 main.py --part three
```

Runs the decoder with ALiBi positional bias.

---

# Hyperparameters

- Batch size: 16
- Block size: 32
- Embedding dimension: 64
- Heads: 2
- Layers: 4
- Feedforward hidden size: 100

---

# Key Technical Highlights

✔ Implemented multi-head attention manually (no `nn.MultiheadAttention`)  
✔ Built causal masking from scratch  
✔ Implemented custom LayerNorm  
✔ Implemented sinusoidal positional encoding  
✔ Implemented ALiBi positional bias  
✔ Attention visualization & normalization check  
✔ Perplexity evaluation pipeline  

---

# What This Demonstrates

- Deep understanding of Transformer internals
- Ability to build LLM components from first principles
- Experience evaluating language models via perplexity
- Understanding of positional encoding design trade-offs
- Ability to analyze generalization behavior across distributions

---

# Future Extensions

- Longer context training
- Relative position bias (T5-style)
- Rotary embeddings (RoPE)
- Larger vocabulary / dataset scaling
- GPT-style generation

---

# Author

Built as part of CSE256 Statistical Natural Language Processing coursework.  
Implemented fully from scratch in PyTorch.

---

If you're interested in discussing Transformer architectures, LLM training, or research collaborations — feel free to reach out!

