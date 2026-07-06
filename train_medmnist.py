#!/usr/bin/env python3

import os
import json
import random
import argparse
from pathlib import Path

import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torchvision.transforms as transforms
from torchvision.models import resnet18

import medmnist
from medmnist import INFO, Evaluator


# -------------------------
# Utils
# -------------------------

def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def make_model(num_classes, in_channels, device):
    model = resnet18(num_classes=num_classes)
    model.conv1 = nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    return model.to(device)


def evaluate(model, loader, data_flag, split, size, device):
    model.eval()
    y_score = []

    with torch.no_grad():
        for inputs, _ in loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            outputs = outputs.softmax(dim=-1)
            y_score.append(outputs.cpu())

    y_score = torch.cat(y_score, dim=0).numpy()
    evaluator = Evaluator(data_flag, split, size=size)
    return evaluator.evaluate(y_score)  # (auc, acc)


# -------------------------
# Main
# -------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_flag", type=str, default="dermamnist")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--save_dir", type=str, default="runs/dermamnist")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--size", type=int, default=28)
    parser.add_argument("--do_test", action="store_true")
    args = parser.parse_args()

    # setup
    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    print("Config:", vars(args))
    print("Device:", device)

    # dataset info
    info = INFO[args.data_flag]
    n_channels = info["n_channels"]
    n_classes = len(info["label"])
    DataClass = getattr(medmnist, info["python_class"])

    # transforms
    mean = [0.5] * n_channels
    std = [0.5] * n_channels
    data_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])

    # datasets
    train_dataset = DataClass(split="train", transform=data_transform, download=args.download, size=args.size)
    val_dataset   = DataClass(split="val",   transform=data_transform, download=args.download, size=args.size)
    test_dataset  = DataClass(split="test",  transform=data_transform, download=args.download, size=args.size)

    # loaders (seeded shuffle)
    g = torch.Generator()
    g.manual_seed(args.seed)

    train_loader = data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, generator=g)
    val_loader   = data.DataLoader(val_dataset,   batch_size=2*args.batch_size, shuffle=False)
    test_loader  = data.DataLoader(test_dataset,  batch_size=2*args.batch_size, shuffle=False)

    # model
    model = make_model(n_classes, n_channels, device)

    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[50, 75], gamma=0.1)

    best_val_acc = -1.0
    best_path = save_dir / "best_model_val.pth"

    # save config
    with open(save_dir / "config.json", "w") as f:
        json.dump(vars(args), f, indent=2)

    # -------------------------
    # Training
    # -------------------------

    for epoch in range(args.epochs):
        model.train()
        for inputs, targets in tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}"):
            inputs = inputs.to(device)
            targets = targets.to(device).squeeze().long()

            optimizer.zero_grad(set_to_none=True)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

        # validation
        val_auc, val_acc = evaluate(model, val_loader, args.data_flag, "val", args.size, device)
        lr_now = optimizer.param_groups[0]["lr"]

        print(f"Epoch {epoch+1:03d} | lr: {lr_now:.1e} | val auc: {val_auc:.3f} | val acc: {val_acc:.3f}")

        if val_acc > best_val_acc:
            best_val_acc = float(val_acc)
            torch.save({
                "model_state": model.state_dict(),
                "epoch": epoch + 1,
                "val_acc": best_val_acc,
                "seed": args.seed
            }, best_path)
            print(f"   New best saved (acc={best_val_acc:.4f})")

        scheduler.step()

    print(f"Training finished. Best val acc = {best_val_acc:.4f}")

    # -------------------------
    # Test
    # -------------------------

    if args.do_test:
        print("\nRunning TEST evaluation with best model...")

        checkpoint = torch.load(best_path, map_location=device)
        model.load_state_dict(checkpoint["model_state"])
        model.to(device)

        test_auc, test_acc = evaluate(model, test_loader, args.data_flag, "test", args.size, device)

        print(f"TEST results:")
        print(f"  auc: {test_auc:.4f}")
        print(f"  acc: {test_acc:.4f}")

        acc_str = f"{test_acc:.4f}"
        final_path = save_dir / f"best_model_val_testacc_{acc_str}.pth"

        torch.save({
            "model_state": model.state_dict(),
            "seed": args.seed,
            "test_auc": float(test_auc),
            "test_acc": float(test_acc),
        }, final_path)

        print(f"Model saved as: {final_path.name}")


if __name__ == "__main__":
    main()

