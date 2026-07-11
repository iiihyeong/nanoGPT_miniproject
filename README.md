# 🧠 nanoGPT Reimplementation from Scratch

> A from-scratch implementation of a **163M-parameter GPT** (GPT-2 Small architecture) in PyTorch, built to understand the Transformer end to end — from tokenization to text generation.

## 📌 Overview

This project rebuilds a GPT-style language model without high-level abstractions, following *Build a Large Language Model (From Scratch)* as a guide and Karpathy's nanoGPT as a reference for cross-checking. Each component was implemented and verified incrementally, from the tokenizer up to a working generation loop.

## 🛠️ What's Implemented

- **Tokenizer & data pipeline** — text encoding/decoding and batching for causal language modeling
- **Self-attention** — scaled dot-product attention with causal masking, built from core tensor operations
- **Multi-head attention** — parallel heads with query/key/value projections and output projection
- **Transformer block** — LayerNorm, GELU, feed-forward network, and residual (shortcut) connections
- **GPT model** — token + positional embeddings and stacked Transformer blocks assembled into a full **163M-parameter** model
- **Text generation** — a greedy decoding loop that autoregressively predicts the next token

## 🚀 How to Run

```bash
git clone https://github.com/iiihyeong/nanoGPT_miniproject.git
cd nanoGPT_miniproject
# Open the chapter notebooks (ch02-ch04) and run the cells top to bottom
jupyter notebook
```

The final notebook (`ch04`) assembles the full model and runs a generation example (e.g. `"Hello, I am"` -> next tokens).

## 💡 What I Learned

Implementing attention by hand made concepts I had only read about in papers concrete. Computing attention scores directly (`queries @ keys.T`) showed how the model measures similarity between tokens, and the final `softmax -> argmax` step made it clear how the model actually *predicts* the next token from those scores. Building the model this way turned the Transformer from an abstract diagram into something I could trace line by line.

---

*Reimplemented as an AI/ML study project. Author: Jaehyeong Choi ([@iiihyeong](https://github.com/iiihyeong))*
