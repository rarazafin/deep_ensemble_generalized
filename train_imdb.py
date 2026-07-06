import re, math, random, os, argparse
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from collections import Counter

# -----------------
# Args
# -----------------
parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--epochs", type=int, default=20)
parser.add_argument("--batch_size", type=int, default=32)
parser.add_argument("--save_dir", type=str, default="deep_ensemble_tu/logs/transformers")
args = parser.parse_args()

SEED = args.seed
EPOCHS = args.epochs
BATCH_SIZE = args.batch_size
BASE_SAVE_DIR = args.save_dir

# -----------------
# Config fixe
# -----------------
CSV_PATH = "./data/IMDB Dataset.csv"
N_TRAIN, N_VAL, N_TEST = 35_000, 7_500, 7_500
MAX_LEN = 256
EMB = 128
NHEAD = 4
NLAYERS = 2
FF = 256
DROPOUT = 0.1
LR = 3e-4

SAVE_DIR = os.path.join(BASE_SAVE_DIR)
BEST_PATH = os.path.join(SAVE_DIR, "best_model.pt")

# -----------------
# Repro
# -----------------
random.seed(0)
torch.manual_seed(SEED)
device = "cuda" if torch.cuda.is_available() else "cpu"

# -----------------
# Tokenizer
# -----------------
def tok(s):
    s = s.lower()
    s = re.sub(r"<br\s*/?>", " ", s)
    s = re.sub(r"[^a-z0-9\s']", " ", s)
    return s.split()

# -----------------
# Load + split
# -----------------
df = pd.read_csv(CSV_PATH)
texts = df["review"].tolist()
labels = (df["sentiment"].str.lower() == "positive").astype(int).tolist()

idx = list(range(len(texts)))
random.shuffle(idx)
idx = idx[: N_TRAIN + N_VAL + N_TEST]
tr_idx = idx[:N_TRAIN]
va_idx = idx[N_TRAIN:N_TRAIN+N_VAL]
te_idx = idx[N_TRAIN+N_VAL:]

tr_texts = [texts[i] for i in tr_idx]; tr_labels = [labels[i] for i in tr_idx]
va_texts = [texts[i] for i in va_idx]; va_labels = [labels[i] for i in va_idx]
te_texts = [texts[i] for i in te_idx]; te_labels = [labels[i] for i in te_idx]

# -----------------
# Vocab
# -----------------
cnt = Counter()
for t in tr_texts:
    cnt.update(tok(t))

PAD, UNK = "<pad>", "<unk>"
itos = [PAD, UNK] + [w for w, _ in cnt.most_common(30_000)]
stoi = {w: i for i, w in enumerate(itos)}
pad_id, unk_id = stoi[PAD], stoi[UNK]

def encode(text):
    ids = [stoi.get(w, unk_id) for w in tok(text)][:MAX_LEN]
    ids += [pad_id] * (MAX_LEN - len(ids))
    return torch.tensor(ids, dtype=torch.long)

class IMDB(Dataset):
    def __init__(self, X, y):
        self.X, self.y = X, y
    def __len__(self): return len(self.X)
    def __getitem__(self, i):
        return encode(self.X[i]), torch.tensor(self.y[i], dtype=torch.long)

g = torch.Generator()
g.manual_seed(SEED)
train_dl = DataLoader(IMDB(tr_texts, tr_labels), batch_size=BATCH_SIZE, shuffle=True, generator=g)
val_dl   = DataLoader(IMDB(va_texts, va_labels), batch_size=BATCH_SIZE)
test_dl  = DataLoader(IMDB(te_texts, te_labels), batch_size=BATCH_SIZE)

# -----------------
# Model
# -----------------
class PosEnc(nn.Module):
    def __init__(self, d, max_len=MAX_LEN):
        super().__init__()
        pe = torch.zeros(max_len, d)
        pos = torch.arange(0, max_len).unsqueeze(1)
        div = torch.exp(torch.arange(0, d, 2) * (-math.log(10000.0) / d))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))
    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

class TransformerSent(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, EMB, padding_idx=pad_id)
        self.pos = PosEnc(EMB)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=EMB, nhead=NHEAD, dim_feedforward=FF,
            dropout=DROPOUT, batch_first=True, activation="gelu"
        )
        self.enc = nn.TransformerEncoder(enc_layer, num_layers=NLAYERS)
        self.drop = nn.Dropout(DROPOUT)
        self.fc = nn.Linear(EMB, 2)

    def forward(self, ids):
        mask = (ids == pad_id)
        x = self.emb(ids)
        x = self.pos(x)
        x = self.enc(x, src_key_padding_mask=mask)
        valid = (~mask).unsqueeze(-1)
        x = (x * valid).sum(1) / valid.sum(1).clamp(min=1)
        return self.fc(self.drop(x))

model = TransformerSent(len(itos)).to(device)
opt = torch.optim.AdamW(model.parameters(), lr=LR)
crit = nn.CrossEntropyLoss()

@torch.no_grad()
def eval_acc(dl):
    model.eval()
    n, ok, loss_sum = 0, 0, 0.0
    for ids, y in dl:
        ids, y = ids.to(device), y.to(device)
        logits = model(ids)
        loss = crit(logits, y)
        loss_sum += loss.item() * y.size(0)
        ok += (logits.argmax(1) == y).sum().item()
        n += y.size(0)
    return loss_sum / n, ok / n

# -----------------
# Train + save best
# -----------------
os.makedirs(SAVE_DIR, exist_ok=True)

best_val_acc = -1.0

for ep in range(1, EPOCHS + 1):
    model.train()
    for ids, y in train_dl:
        ids, y = ids.to(device), y.to(device)
        opt.zero_grad()
        loss = crit(model(ids), y)
        loss.backward()
        opt.step()

    tr_loss, tr_acc = eval_acc(train_dl)
    va_loss, va_acc = eval_acc(val_dl)
    print(f"[seed {SEED}] epoch {ep}: train {tr_acc:.3f} | val {va_acc:.3f}")

    if va_acc > best_val_acc:
        best_val_acc = va_acc
        torch.save(model.state_dict(), BEST_PATH)
        print(f"  -> saved best model to {BEST_PATH}")

# -----------------
# Test best
# -----------------
model.load_state_dict(torch.load(BEST_PATH, map_location=device))
_, te_acc = eval_acc(test_dl)
print(f"[seed {SEED}] FINAL TEST ACC: {te_acc:.4f}")

