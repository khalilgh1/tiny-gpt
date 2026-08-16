# tiny-gpt

A minimal implementation of a GPT2-like language model from scratch.

This project explores the core mechanics of transformer-based language modeling through a from-scratch build. It starts with simple bigram models and gradually introduces the essential pieces: token and positional embeddings, self-attention, multi-head attention, feed-forward networks, and layer normalization. The model is trained on character-level text data and learns to generate coherent sequences by predicting the next token given a context window.

## What's included

- Bigram language model (baseline)
- Average pooling language model with positional awareness
- Multi-head self-attention mechanism  
- Full transformer decoder blocks
- Character-level tokenization and generation
- Training and evaluation loops

## Usage

The main implementation lives in `gpt_dev.ipynb`. You can run the notebook to see the model being built step by step, train it on the included Shakespeare text, and generate samples.

```python
# Generate text from a trained model
idx = torch.zeros((1, 1), dtype=torch.long)
generated = model.generate(idx, max_new_tokens=100)
print(decode(generated[0].tolist()))
```

---

Built following [Andrej Karpathy](https://karpathy.ai/)'s *Zero To Hero* tutorial series on neural networks.
