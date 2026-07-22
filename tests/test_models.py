"""Model smoke tests on tiny tensors: shapes, learning, determinism."""

import numpy as np
import torch

from geoverdict import config as cfg
from geoverdict import models as M


def tiny_data(n=64, c=6, hw=32, seed=0):
    """Separable toy pairs: positives lose 'NIR' and gain 'SWIR' from T1 to T2 —
    the actual physics of clearing, so a learning test is also a sanity test."""
    rng = np.random.default_rng(seed)
    x1 = rng.normal(0.25, 0.05, (n, c, hw, hw)).astype(np.float32)
    x2 = x1 + rng.normal(0, 0.01, x1.shape).astype(np.float32)
    y = (rng.uniform(size=n) < 0.5).astype(np.float32)
    x2[y == 1, 3] -= 0.15   # NIR collapse
    x2[y == 1, 5] += 0.10   # SWIR2 rise
    return x1, x2, y


def test_forward_shape():
    m = M.SiameseChangeNet()
    z = m(torch.zeros(4, 6, 32, 32), torch.zeros(4, 6, 32, 32))
    assert z.shape == (4,)


def test_learns_separable_change():
    cfg.set_seed(0)
    x1, x2, y = tiny_data(n=128)
    ds = M.ChipPairDataset(x1, x2, y, augment=False)
    loader = torch.utils.data.DataLoader(ds, batch_size=32, shuffle=True)
    model = M.SiameseChangeNet(width=8)
    hist = M.fit(model, loader, loader, epochs=8, lr=2e-3, device="cpu", patience=8)
    assert hist.best_val_pr_auc > 0.9, f"failed to learn a separable toy problem: {hist.best_val_pr_auc}"


def test_augmentation_moves_both_dates_together():
    x1, x2, y = tiny_data(n=4)
    x1[:, :, :16, :] = 0.0   # spatial marker in the top half of T1
    x2[:, :, :16, :] = 0.0   # ... and T2
    ds = M.ChipPairDataset(x1, x2, y, augment=True, seed=3)
    a, b, _ = ds[0]
    # wherever the marker went, it must have gone to the SAME place in both
    assert torch.equal(a[0] == 0, b[0] == 0)


def test_dataset_normalisation():
    x1, x2, y = tiny_data()
    stats = {"mean": x1.mean(axis=(0, 2, 3)), "std": x1.std(axis=(0, 2, 3))}
    ds = M.ChipPairDataset(x1, x2, y, stats=stats)
    assert abs(float(ds.x1.mean())) < 0.05
