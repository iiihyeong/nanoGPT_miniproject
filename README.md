🧠 nanoGPT Reimplementation from Scratch


A from-scratch reimplementation of a 163M-parameter GPT in PyTorch, built to understand the Transformer architecture end to end — from tokenization to multi-head self-attention.



📌 Overview

This project rebuilds the core components of a GPT-style model without relying on high-level abstractions, as part of a study bridging statistical theory and hands-on implementation. Each building block is implemented and verified incrementally.

🛠️ Implemented Components


Tokenizer & data pipeline — text encoding/decoding and batching for causal language modeling
Self-attention — scaled dot-product attention built from core tensor operations
Causal masking — autoregressive masking so each token attends only to past tokens
Multi-head attention — parallel attention heads combined into a single projection
GPT model — Transformer blocks (attention + FFN + LayerNorm + residuals) assembled into a full 163M-parameter model

Reimplemented as part of an AI/ML study. Author: Jaehyeong Choi (@iiihyeong)
