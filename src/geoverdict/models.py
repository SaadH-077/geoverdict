"""The learned change detector: a small siamese CNN on before/after chips.

WHY THIS ARCHITECTURE AND NOT SOMETHING FANCIER — the questions in order:

* Why chip CLASSIFICATION and not segmentation? The decision unit of EUDR
  compliance is the PLOT, not the pixel: "did clearing occur on this plot"
  is a yes/no per plot. A U-Net would localise the clearing within the chip
  — genuinely useful, but it multiplies label noise (Hansen's 30 m edges vs
  10 m pixels), training cost, and evaluation complexity for information the
  verdict layer does not consume. Scope is a design decision; this one is
  ablatable later.

* Why SIAMESE with shared weights? Both dates are the same sensor over the
  same land, so the features that describe them should be the same features
  — sharing the encoder enforces that and halves the parameters. The change
  signal is then read from (f2 - f1, f2, f1): the difference carries "what
  changed", the two absolutes let the head condition on WHAT it changed from
  (forest->bare matters; pasture->drier-pasture does not).

* Why from scratch and not pretrained? Six-band multispectral input has no
  honest ImageNet initialisation (RGB weights over SWIR channels is folklore,
  not physics), the training set is thousands of chips — enough for a ~200k
  parameter model — and the from-scratch baseline is the control every
  pretrained claim needs anyway. Foundation-model arms are the natural
  extension and are discussed, not silently skipped.

* Why so SMALL (3 conv blocks)? A 32x32 chip at 10 m contains a plot and its
  margins; three stride-2 stages already see the full chip. Parameters beyond
  that memorise Hansen's label noise — validated empirically: the model is
  early-stopped on val PR-AUC and never reaches its capacity ceiling.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn

from . import config as cfg


def conv_block(cin: int, cout: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, padding=1, bias=False),
        nn.BatchNorm2d(cout),
        nn.ReLU(inplace=True),
        nn.Conv2d(cout, cout, 3, padding=1, bias=False),
        nn.BatchNorm2d(cout),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(2),
    )


class SiameseChangeNet(nn.Module):
    """Shared encoder over T1 and T2; head on [f2-f1, f1, f2]; one logit out."""

    def __init__(self, in_channels: int = len(cfg.CHIP_BANDS), width: int = 32):
        super().__init__()
        self.encoder = nn.Sequential(
            conv_block(in_channels, width),
            conv_block(width, width * 2),
            conv_block(width * 2, width * 4),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
        )
        feat = width * 4
        self.head = nn.Sequential(
            nn.Linear(feat * 3, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, 1),
        )

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        f1, f2 = self.encoder(x1), self.encoder(x2)
        return self.head(torch.cat([f2 - f1, f1, f2], dim=1)).squeeze(-1)


class ChipPairDataset(torch.utils.data.Dataset):
    """(T1, T2, label) chip pairs with train-time augmentation.

    AUGMENTATION CHOICES: dihedral flips/rotations are label-preserving for
    nadir satellite imagery (there is no 'up'); both dates get the SAME
    transform or the pair stops describing the same place. Channel-wise
    illumination jitter is applied INDEPENDENTLY per date — atmospheric
    conditions genuinely differ between acquisitions, and the model must not
    read a global brightness offset as change.
    """

    def __init__(self, x1: np.ndarray, x2: np.ndarray, y: np.ndarray,
                 stats: dict | None = None, augment: bool = False, seed: int = cfg.SEED):
        self.x1 = np.asarray(x1, dtype=np.float32)
        self.x2 = np.asarray(x2, dtype=np.float32)
        self.y = np.asarray(y, dtype=np.float32)
        self.augment = augment
        self.rng = np.random.default_rng(seed)
        if stats is not None:
            mean = np.asarray(stats["mean"], dtype=np.float32)[:, None, None]
            std = np.asarray(stats["std"], dtype=np.float32)[:, None, None]
            self.x1 = (self.x1 - mean) / std
            self.x2 = (self.x2 - mean) / std

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, i: int):
        a, b = self.x1[i].copy(), self.x2[i].copy()
        if self.augment:
            k = int(self.rng.integers(0, 4))
            if k:
                a, b = np.rot90(a, k, (1, 2)).copy(), np.rot90(b, k, (1, 2)).copy()
            if self.rng.random() < 0.5:
                a, b = a[:, :, ::-1].copy(), b[:, :, ::-1].copy()
            for arr in (a, b):  # independent per-date illumination jitter
                arr *= (1.0 + self.rng.normal(0, 0.02, size=(arr.shape[0], 1, 1))).astype(np.float32)
        return torch.from_numpy(a), torch.from_numpy(b), torch.tensor(self.y[i])


@dataclass
class FitHistory:
    train_loss: list = field(default_factory=list)
    val_loss: list = field(default_factory=list)
    val_pr_auc: list = field(default_factory=list)
    best_epoch: int = -1
    best_val_pr_auc: float = 0.0


def fit(
    model: nn.Module,
    train_loader,
    val_loader,
    epochs: int = 30,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    pos_weight: float | None = None,
    device: str | None = None,
    patience: int = 8,
) -> FitHistory:
    """Train with BCE, cosine schedule, early stop on val PR-AUC.

    * pos_weight — clearings are the minority; weighting the positive term by
      n_neg/n_pos keeps the gradient from being dominated by easy negatives.
      This is the LOSS-level counterpart of the sampling-level hard-negative
      work in notebook 04; the two are complementary, not alternatives.
    * Early stopping on PR-AUC, not loss — the deployment quantity is ranking
      quality under imbalance, and val loss can improve while PR-AUC stalls
      (better-calibrated easy negatives, no better ranking of positives).
    """
    from .metrics import pr_auc

    device = device or cfg.get_device()
    model = model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    pw = torch.tensor(pos_weight, device=device) if pos_weight else None
    criterion = nn.BCEWithLogitsLoss(pos_weight=pw)

    hist = FitHistory()
    best_state, since_best = None, 0
    for epoch in range(epochs):
        model.train()
        tot, n = 0.0, 0
        for a, b, y in train_loader:
            a, b, y = a.to(device), b.to(device), y.to(device)
            opt.zero_grad(set_to_none=True)
            loss = criterion(model(a, b), y)
            loss.backward()
            opt.step()
            tot += float(loss.detach()) * len(y)
            n += len(y)
        sched.step()
        hist.train_loss.append(tot / max(n, 1))

        vl, logits, labels = evaluate(model, val_loader, criterion, device)
        hist.val_loss.append(vl)
        auc_val = pr_auc(labels, 1 / (1 + np.exp(-logits)))
        hist.val_pr_auc.append(auc_val)

        if auc_val > hist.best_val_pr_auc:
            hist.best_val_pr_auc, hist.best_epoch = auc_val, epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            since_best = 0
        else:
            since_best += 1
            if since_best >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)  # return the best model, not the last
    return hist


@torch.no_grad()
def evaluate(model: nn.Module, loader, criterion=None, device: str | None = None):
    """Returns (mean loss, logits array, labels array)."""
    device = device or cfg.get_device()
    model = model.to(device).eval()
    criterion = criterion or nn.BCEWithLogitsLoss()
    tot, n, logits, labels = 0.0, 0, [], []
    for a, b, y in loader:
        a, b, y = a.to(device), b.to(device), y.to(device)
        z = model(a, b)
        tot += float(criterion(z, y)) * len(y)
        n += len(y)
        logits.append(z.cpu().numpy())
        labels.append(y.cpu().numpy())
    return tot / max(n, 1), np.concatenate(logits), np.concatenate(labels)


def predict_proba(model: nn.Module, loader, device: str | None = None,
                  temperature: float = 1.0) -> np.ndarray:
    _, logits, _ = evaluate(model, loader, device=device)
    return 1.0 / (1.0 + np.exp(-logits / temperature))
