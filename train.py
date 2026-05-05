import torch
import torch.nn as nn
import torch.optim as optim
from collections import Counter
from datasets import load_dataset
from sklearn.datasets import fetch_20newsgroups
import random

# ======================
# 1. Tokenizer
# ======================
def build_vocab(texts, max_size=20000):
    counter = Counter()
    for t in texts:
        counter.update(t.lower().split())
    vocab = {w: i+1 for i, (w, _) in enumerate(counter.most_common(max_size))}
    return vocab

def encode(text, vocab, max_len=200):
    tokens = [vocab.get(w, 0) for w in text.lower().split()]
    tokens = tokens[:max_len]
    tokens += [0] * (max_len - len(tokens))
    return torch.tensor(tokens)

# ======================
# 2. Model
# ======================
class SimpleModel(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_classes):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.conv = nn.Conv1d(embed_dim, 128, kernel_size=3)
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(128, 64)
        self.fc2 = nn.Linear(64, num_classes)

    def forward(self, x):
        x = self.embedding(x)          # (B, L, E)
        x = x.permute(0, 2, 1)         # (B, E, L)
        x = self.conv(x)
        x = self.relu(x)
        x = torch.max(x, dim=2).values
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

# ======================
# 3. Trening
# ======================
def train(model, X, y, epochs=3):
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        total_loss = 0

        for i in range(len(X)):
            x = X[i].unsqueeze(0)
            label = torch.tensor([y[i]])

            pred = model(x)
            loss = loss_fn(pred, label)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch+1}, loss={total_loss:.4f}")

# ======================
# 4. IMDB
# ======================
def run_imdb():
    print("Loading IMDB...")
    imdb = load_dataset("imdb")

    texts = imdb["train"]["text"][:2000]
    labels = imdb["train"]["label"][:2000]

    vocab = build_vocab(texts)

    X = [encode(t, vocab) for t in texts]
    y = labels

    model = SimpleModel(len(vocab)+1, 100, 2)

    train(model, X, y)

# ======================
# 5. 20 Newsgroups
# ======================
def run_newsgroups():
    print("Loading 20 Newsgroups...")
    data = fetch_20newsgroups(subset='train')

    texts = data.data[:2000]
    labels = data.target[:2000]

    vocab = build_vocab(texts)

    X = [encode(t, vocab) for t in texts]
    y = labels

    model = SimpleModel(len(vocab)+1, 100, 20)

    train(model, X, y)

# ======================
# 6. MAIN
# ======================
if __name__ == "__main__":
    print("1 - IMDB")
    print("2 - 20 Newsgroups")
    choice = input("Wybierz dataset: ")

    if choice == "1":
        run_imdb()
    else:
        run_newsgroups()