import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from collections import Counter
from datasets import load_dataset
from sklearn.datasets import fetch_20newsgroups
from sklearn.model_selection import train_test_split
import pandas as pd

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# 1. Tokenizer / vocab


def build_vocab(texts, max_size=20000, min_freq=2):
    counter = Counter()
    for t in texts:
        counter.update(t.lower().split())
    vocab = {"<pad>": 0, "<unk>": 1}
    for w, c in counter.most_common(max_size):
        if c < min_freq:
            continue
        vocab[w] = len(vocab)
    return vocab

def encode(text, vocab, max_len=200):
    tokens = [vocab.get(w, 1) for w in text.lower().split()]
    tokens = tokens[:max_len]
    tokens += [0] * (max_len - len(tokens))
    return torch.tensor(tokens, dtype=torch.long)


# 2. Dataset


class TextDataset(Dataset):
    def __init__(self, texts, labels, vocab, max_len=200):
        self.X = [encode(t, vocab, max_len) for t in texts]
        self.y = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# 3. Model: Conv1D + Linear + Linear


class SimpleCNN(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_classes,
                 num_conv_layers=1, activation_fn=nn.ReLU):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embed_dim)

        layers = []
        in_channels = embed_dim

        for _ in range(num_conv_layers):
            layers.append(nn.Conv1d(in_channels, 128, kernel_size=3, padding=1))
            layers.append(activation_fn())
            in_channels = 128

        self.conv = nn.Sequential(*layers)

        self.fc1 = nn.Linear(128, 64)
        self.act = activation_fn()
        self.fc2 = nn.Linear(64, num_classes)

    def forward(self, x):
        x = self.embedding(x)          # (B, L, E)
        x = x.permute(0, 2, 1)         # (B, E, L)
        x = self.conv(x)               # (B, 128, L)
        x = torch.max(x, dim=2).values # global max pooling
        x = self.act(self.fc1(x))
        x = self.fc2(x)
        return x


# 4. Accuracy


def compute_accuracy(model, loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for X, y in loader:
            X, y = X.to(device), y.to(device)
            preds = torch.argmax(model(X), dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)
    return correct / total


# 5. Training


def train_model(model, train_loader, val_loader=None, epochs=5):
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        model.train()
        total_loss = 0

        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            logits = model(X)
            loss = loss_fn(logits, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * y.size(0)

        msg = f"Epoch {epoch+1}, loss={total_loss/len(train_loader.dataset):.4f}"

        if val_loader:
            acc = compute_accuracy(model, val_loader)
            msg += f", val_acc={acc:.4f}"

        print(msg)


# 6. IMDB


def run_imdb(num_conv_layers, activation_fn):
    print("Loading IMDB...")
    imdb = load_dataset("imdb")

    texts = imdb["train"]["text"]
    labels = imdb["train"]["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        texts, labels, test_size=0.2, random_state=42
    )

    vocab = build_vocab(X_train)

    train_ds = TextDataset(X_train, y_train, vocab)
    test_ds  = TextDataset(X_test,  y_test,  vocab)

    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
    test_loader  = DataLoader(test_ds,  batch_size=64)

    model = SimpleCNN(
        vocab_size=len(vocab),
        embed_dim=100,
        num_classes=2,
        num_conv_layers=num_conv_layers,
        activation_fn=activation_fn
    )

    print("Training IMDB model...")
    train_model(model, train_loader, val_loader=test_loader, epochs=5)

    acc = compute_accuracy(model, test_loader)
    print(f"IMDB Test Accuracy: {acc:.4f}")
    return acc


# 7. 20 Newsgroups


def run_newsgroups(num_conv_layers, activation_fn):
    print("Loading 20 Newsgroups...")
    data = fetch_20newsgroups(subset='train')

    texts = data.data
    labels = data.target

    X_train, X_test, y_train, y_test = train_test_split(
        texts, labels, test_size=0.2, random_state=42
    )

    vocab = build_vocab(X_train)

    train_ds = TextDataset(X_train, y_train, vocab)
    test_ds  = TextDataset(X_test,  y_test,  vocab)

    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
    test_loader  = DataLoader(test_ds,  batch_size=64)

    model = SimpleCNN(
        vocab_size=len(vocab),
        embed_dim=100,
        num_classes=20,
        num_conv_layers=num_conv_layers,
        activation_fn=activation_fn
    )

    print("Training 20 Newsgroups model...")
    train_model(model, train_loader, val_loader=test_loader, epochs=5)

    acc = compute_accuracy(model, test_loader)
    print(f"20 Newsgroups Test Accuracy: {acc:.4f}")
    return acc


# 8. Main Experiment


def experiment():
    results = []

    configs = [
        (1, nn.ReLU),
        (2, nn.ReLU),
        (3, nn.ReLU),
        (1, nn.LeakyReLU),
        (1, nn.GELU),
    ]

    for num_layers, act in configs:
        print(f"\nRunning IMDB: {num_layers} conv layers, {act.__name__}")
        acc = run_imdb(num_layers, act)
        results.append({
            "Dataset": "IMDB",
            "Conv Layers": num_layers,
            "Activation": act.__name__,
            "Accuracy": acc
        })

    for num_layers, act in configs:
        print(f"\nRunning 20NG: {num_layers} conv layers, {act.__name__}")
        acc = run_newsgroups(num_layers, act)
        results.append({
            "Dataset": "20NG",
            "Conv Layers": num_layers,
            "Activation": act.__name__,
            "Accuracy": acc
        })

    df = pd.DataFrame(results)
    print("\nFINAL RESULTS:")
    print(df)


# 9. MAIN


if __name__ == "__main__":
    experiment()
