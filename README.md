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

## Model Statistics

- **Dataset**: 1,115,393 characters from Shakespeare text
- **Vocabulary**: 65 unique characters (letters, punctuation, numbers)
- **Architecture**: 6 transformer blocks with 6 attention heads
- **Parameters**: 10.79M trainable parameters
- **Embedding dimension**: 384
- **Context window**: 256 tokens

## Training Results

The model was trained for 5,000 iterations with evaluation every 500 steps:

| Step | Train Loss | Val Loss |
|------|-----------|----------|
| 0    | 4.2227    | 4.2305   |
| 500  | 1.7204    | 1.8783   |
| 1000 | 1.3477    | 1.5828   |
| 2000 | 1.1273    | 1.5182   |
| 3000 | 0.9748    | 1.5410   |
| 4500 | 0.7431    | 1.6973   |

The model converges quickly, with the training loss dropping 85% from initialization. The model learns to generate Shakespeare-like text after training (~31 minutes on GPU).

## Usage

The main implementation lives in `gpt_dev.ipynb`. You can run the notebook to see the model being built step by step, train it on the included Shakespeare text, and generate samples.

```python
# Generate text from a trained model
idx = torch.zeros((1, 1), dtype=torch.long)
generated = model.generate(idx, max_new_tokens=100)
print(decode(generated[0].tolist()))
```

Sample output (500 tokens):
```
yire to the merital mercy of the gate,
To comfort our hope the nay rest: but when I was are
A pretty general gentlewoman wise, and they are
Turn for't.

ANGELO:
Why prove itself he that respects but we stands,
Being ne'er the vent.
```

**Observations**: The generated text exhibits proper English vocabulary and maintains Shakespeare-like structure (character names, dramatic dialogue). However, the semantic meaning is largely incoherent—a common trait of character-level language models trained with limited context. The model learns character patterns and formatting rules but struggles to maintain consistent meaning across long sequences, illustrating the gap between surface-level pattern matching and genuine language understanding.

---

Built following [Andrej Karpathy](https://karpathy.ai/)'s *Zero To Hero* tutorial series on neural networks.
