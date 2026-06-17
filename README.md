
# Character-Level Language Model (Azerbaijani Text)
 
This project was built to learn how transformer architectures work internally. The code is **not originally mine** — it follows [Andrej Karpathy's "Let's build GPT"](https://www.youtube.com/watch?v=kCc8FmEb1nY) video and was hand-typed, run, and studied step by step as part of the learning process.
 
The goal is simple: start from the most primitive language model (a bigram model) and gradually build up to a full transformer (GPT-style) model, in order to understand self-attention, multi-head attention, and transformer blocks from the ground up.
 
## What's in this repo
 
| File | Description |
|------|--------------|
| `main.ipynb` | A simple **bigram language model** with no transformer. A notebook used to walk through tokenization, data preparation, and the simplest possible model based on `nn.Embedding`. |
| `main.py` | A **transformer-based (GPT-style)** language model. A full mini-GPT implementation including self-attention heads, multi-head attention, a feed-forward network, and transformer blocks. |
| `text.txt` | The training dataset — a synthetic Azerbaijani text corpus made of fictional letters, short stories, proverbs, dialogues, and poems. |
 
## Model 1 — Bigram Language Model (`main.ipynb`)
 
This is the simplest possible language model, built without any transformer components:
 
- The text is tokenized at the character level (vocabulary size: 72 unique characters).
- Each character is mapped to an integer (`stoi` / `itos`).
- The model consists of nothing more than an `nn.Embedding` table — each character directly predicts the probability distribution of the next character, with no context at all (a classic bigram model).
- The goal here was to understand tokenization, batch preparation (context/target pairs), and how the training loop (loss, backward, optimizer step) works.
This model has no memory — it only predicts based on the single previous character, so the generated text ends up grammatically meaningless.
 
## Model 2 — GPT-style Transformer (`main.py`)
 
This file implements a small but fully functional transformer model:
 
- **Self-Attention Head** — computes key, query, and value vectors, with causal masking (`tril`) so each position can only attend to past tokens.
- **Multi-Head Attention** — combines several attention heads running in parallel.
- **Feed-Forward** — a simple MLP applied independently at each position.
- **Transformer Block** — combines attention and feed-forward layers with residual connections and layer normalization.
- **Positional Embedding** — added so the model can understand token order in the sequence.
### Hyperparameters
 
```
batch_size   = 16
block_size   = 32
n_embd       = 64
n_head       = 4
n_layer      = 4
max_iters    = 5000
learning_rate = 1e-3
```
 
The model has only **0.21M parameters** and was trained for 5000 steps.
 
### Results
 
Loss decreased steadily during training:
 
```
step 0:    train loss 4.2918, val loss 4.2929
step 1000: train loss 0.8648, val loss 0.8795
step 2000: train loss 0.4666, val loss 0.4788
step 3000: train loss 0.3961, val loss 0.4076
step 4999: train loss 0.3480, val loss 0.3634
```
 
The model now generates text that resembles Azerbaijani grammar, dialogue formatting, and even proverb style, instead of pure noise:
 
```
Bir varmış.
Quba-dan sənə dekab başqa sevgi bayını cilə əbidir...
Günel: sənin sən deyirsən ki, mütləq həqiqət yoxdu.
Günel qalib gəlibmiş.
Kənd camaatı sevinibmiş. Lalə ilə dəyişir. Bu təbiidir.
...
```
 
The text is still not fully coherent (names and events get mixed up), but the improvement in syntax, punctuation, and morphology is clear — a good demonstration of how parameters like `vocab_size`, `n_embd`, and `n_layer` affect the final output.
 
## Bigram vs Transformer comparison
 
| | Bigram Model | Transformer Model |
|---|---|---|
| Context | Only the previous 1 token | Up to `block_size` (32 tokens) |
| Mechanism | Simple embedding lookup | Self-attention + feed-forward |
| Final loss | ~2.46 | ~0.35 |
| Output quality | Essentially random characters | Grammatically meaningful sentences |
 
## Usage
 
```bash
pip install torch
 
python main.py
```
 
`main.ipynb` is meant to be run step by step in Jupyter Notebook or VS Code to see how the bigram model works in detail.
 
## Note
 
This is a learning project. The core logic and architecture follow Andrej Karpathy's nanoGPT / "Let's build GPT" material; my own contribution was working through the code line by line, running it, testing it on different (Azerbaijani-language) data, and analyzing the results.
 





