import argparse
import math
import os
import random
import re
from collections import Counter

import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset


CSV_PATH = "./data/IMDB Dataset.csv"
N_TRAIN, N_VAL, N_TEST = 35_000, 7_500, 7_500
MAX_LEN = 256
EMB = 128
NHEAD = 4
NLAYERS = 2
FF = 256
DROPOUT = 0.1
LR = 3e-4


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--save_dir", type=str, default="deep_ensemble_tu/logs/transformers")
    return parser.parse_args()


def seed_everything(seed):
    random.seed(0)
    torch.manual_seed(seed)


def tokenize(text):
    text = text.lower()
    text = re.sub(r"<br\s*/?>", " ", text)
    text = re.sub(r"[^a-z0-9\s']", " ", text)
    return text.split()


def make_split(texts, labels):
    idx = list(range(len(texts)))
    random.shuffle(idx)
    idx = idx[: N_TRAIN + N_VAL + N_TEST]

    train_idx = idx[:N_TRAIN]
    val_idx = idx[N_TRAIN:N_TRAIN + N_VAL]
    test_idx = idx[N_TRAIN + N_VAL:]

    train_texts = [texts[i] for i in train_idx]
    train_labels = [labels[i] for i in train_idx]

    val_texts = [texts[i] for i in val_idx]
    val_labels = [labels[i] for i in val_idx]

    test_texts = [texts[i] for i in test_idx]
    test_labels = [labels[i] for i in test_idx]

    return train_texts, train_labels, val_texts, val_labels, test_texts, test_labels


def make_vocab(texts, max_words=30_000):
    counter = Counter()

    for text in texts:
        counter.update(tokenize(text))

    itos = ["<pad>", "<unk>"] + [word for word, _ in counter.most_common(max_words)]
    stoi = {word: i for i, word in enumerate(itos)}

    return itos, stoi


class IMDBDataset(Dataset):
    def __init__(self, texts, labels, stoi, pad_id, unk_id):
        self.texts = texts
        self.labels = labels
        self.stoi = stoi
        self.pad_id = pad_id
        self.unk_id = unk_id

    def __len__(self):
        return len(self.texts)

    def encode(self, text):
        ids = [self.stoi.get(word, self.unk_id) for word in tokenize(text)][:MAX_LEN]
        ids += [self.pad_id] * (MAX_LEN - len(ids))
        return torch.tensor(ids, dtype=torch.long)

    def __getitem__(self, index):
        ids = self.encode(self.texts[index])
        label = torch.tensor(self.labels[index], dtype=torch.long)
        return ids, label


def make_dataloaders(args, train_texts, train_labels, val_texts, val_labels, test_texts, test_labels, stoi, pad_id, unk_id):
    generator = torch.Generator()
    generator.manual_seed(args.seed)

    train_loader = DataLoader(
        IMDBDataset(train_texts, train_labels, stoi, pad_id, unk_id),
        batch_size=args.batch_size,
        shuffle=True,
        generator=generator,
    )
    val_loader = DataLoader(
        IMDBDataset(val_texts, val_labels, stoi, pad_id, unk_id),
        batch_size=args.batch_size,
    )
    test_loader = DataLoader(
        IMDBDataset(test_texts, test_labels, stoi, pad_id, unk_id),
        batch_size=args.batch_size,
    )

    return train_loader, val_loader, test_loader


class PositionalEncoding(nn.Module):
    def __init__(self, dim, max_len=MAX_LEN):
        super().__init__()

        pe = torch.zeros(max_len, dim)
        pos = torch.arange(0, max_len).unsqueeze(1)
        div = torch.exp(torch.arange(0, dim, 2) * (-math.log(10000.0) / dim))

        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)

        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


class TransformerSentiment(nn.Module):
    def __init__(self, vocab_size, pad_id):
        super().__init__()

        self.pad_id = pad_id
        self.emb = nn.Embedding(vocab_size, EMB, padding_idx=pad_id)
        self.pos = PositionalEncoding(EMB)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=EMB,
            nhead=NHEAD,
            dim_feedforward=FF,
            dropout=DROPOUT,
            batch_first=True,
            activation="gelu",
        )

        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=NLAYERS)
        self.dropout = nn.Dropout(DROPOUT)
        self.fc = nn.Linear(EMB, 2)

    def forward(self, ids):
        mask = ids == self.pad_id

        x = self.emb(ids)
        x = self.pos(x)
        x = self.encoder(x, src_key_padding_mask=mask)

        valid = (~mask).unsqueeze(-1)
        x = (x * valid).sum(1) / valid.sum(1).clamp(min=1)

        return self.fc(self.dropout(x))


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()

    n = 0
    correct = 0
    loss_sum = 0.0

    for ids, labels in loader:
        ids = ids.to(device)
        labels = labels.to(device)

        logits = model(ids)
        loss = criterion(logits, labels)

        loss_sum += loss.item() * labels.size(0)
        correct += (logits.argmax(1) == labels).sum().item()
        n += labels.size(0)

    return loss_sum / n, correct / n


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()

    for ids, labels in loader:
        ids = ids.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        loss = criterion(model(ids), labels)
        loss.backward()
        optimizer.step()


def main():
    args = parse_args()
    seed_everything(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    save_dir = os.path.join(args.save_dir)
    best_path = os.path.join(save_dir, "best_model.pt")

    os.makedirs(save_dir, exist_ok=True)

    df = pd.read_csv(CSV_PATH)
    texts = df["review"].tolist()
    labels = (df["sentiment"].str.lower() == "positive").astype(int).tolist()

    train_texts, train_labels, val_texts, val_labels, test_texts, test_labels = make_split(texts, labels)

    itos, stoi = make_vocab(train_texts)
    pad_id = stoi["<pad>"]
    unk_id = stoi["<unk>"]

    train_loader, val_loader, test_loader = make_dataloaders(
        args,
        train_texts,
        train_labels,
        val_texts,
        val_labels,
        test_texts,
        test_labels,
        stoi,
        pad_id,
        unk_id,
    )

    model = TransformerSentiment(len(itos), pad_id).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

    best_val_acc = -1.0

    for epoch in range(1, args.epochs + 1):
        train_one_epoch(model, train_loader, criterion, optimizer, device)

        train_loss, train_acc = evaluate(model, train_loader, criterion, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)

        print(f"[seed {args.seed}] epoch {epoch}: train {train_acc:.3f} | val {val_acc:.3f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), best_path)
            print(f"  -> saved best model to {best_path}")

    model.load_state_dict(torch.load(best_path, map_location=device))
    _, test_acc = evaluate(model, test_loader, criterion, device)

    print(f"[seed {args.seed}] FINAL TEST ACC: {test_acc:.4f}")


if __name__ == "__main__":
    main()