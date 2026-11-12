"""Train the defect classifier (ResNet18 fine-tune) on an ImageFolder dataset:

    data/train/{clean,corrosion,crack,dent}/*.png
    data/val/{clean,corrosion,crack,dent}/*.png
"""
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from detector import IMG_MEAN, IMG_STD, build_model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--out", default="models/defect_cnn.pt")
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--img", type=int, default=128)
    ap.add_argument("--scratch", action="store_true", help="do not use ImageNet weights")
    ap.add_argument("--workers", type=int, default=2)
    a = ap.parse_args()

    norm = transforms.Normalize(IMG_MEAN.tolist(), IMG_STD.tolist())
    train_tf = transforms.Compose([
        transforms.Resize((a.img, a.img)),
        transforms.RandomHorizontalFlip(), transforms.RandomVerticalFlip(),
        transforms.ColorJitter(0.2, 0.2), transforms.ToTensor(), norm])
    val_tf = transforms.Compose([transforms.Resize((a.img, a.img)), transforms.ToTensor(), norm])

    train_ds = datasets.ImageFolder(Path(a.data) / "train", train_tf)
    val_ds = datasets.ImageFolder(Path(a.data) / "val", val_tf)
    classes = train_ds.classes
    print("classes:", classes, "| train:", len(train_ds), "val:", len(val_ds))
    train_dl = DataLoader(train_ds, a.batch, shuffle=True, num_workers=a.workers)
    val_dl = DataLoader(val_ds, a.batch, num_workers=a.workers)

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = build_model(len(classes), pretrained=not a.scratch).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.epochs)
    loss_fn = nn.CrossEntropyLoss(label_smoothing=0.05)

    best, Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    for ep in range(1, a.epochs + 1):
        model.train()
        total = 0.0
        for x, y in train_dl:
            x, y = x.to(dev), y.to(dev)
            opt.zero_grad()
            loss = loss_fn(model(x), y)
            loss.backward()
            opt.step()
            total += loss.item() * len(x)
        sched.step()

        model.eval()
        conf = np.zeros((len(classes), len(classes)), int)
        with torch.no_grad():
            for x, y in val_dl:
                pred = model(x.to(dev)).argmax(1).cpu()
                for t, p in zip(y, pred):
                    conf[int(t), int(p)] += 1
        acc = conf.trace() / conf.sum()
        print(f"epoch {ep}/{a.epochs}  loss {total / len(train_ds):.4f}  val acc {acc:.3f}")
        if acc >= best:
            best = acc
            torch.save({"state_dict": model.state_dict(), "classes": classes, "img_size": a.img}, a.out)

    print(f"\nBest val acc {best:.3f} -> {a.out}")
    print("Confusion matrix (rows = true, cols = predicted), last epoch:")
    print(classes)
    print(conf)


if __name__ == "__main__":
    main()
