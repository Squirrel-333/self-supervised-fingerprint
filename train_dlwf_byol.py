import os
import json
import copy
import math
import random
import argparse
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report

from dlwf_dataloader import load_dlwf_npz


# --------------------------------------------------
# Paths
# --------------------------------------------------

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_PATH_DEFAULT = os.path.join(
    PROJECT_ROOT,
    "data",
    "closed_world",
    "tor_100w_2500tr.npz",
)


# --------------------------------------------------
# Utilities
# --------------------------------------------------


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)



def get_device(gpu: int) -> torch.device:
    if gpu < 0 or not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(f"cuda:{gpu}")



def save_json(path: str, obj) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


# --------------------------------------------------
# Data preparation
# --------------------------------------------------


def prepare_dlwf_data(
    npz_path,
    normalize=False,
    num_classes=None,
    max_samples=None,
    random_state=42,
):
    """
    Load DLWF data using the shared loader, then optionally:
    1. keep only the first K classes
    2. subsample to max_samples

    Returns:
        X (np.ndarray): shape (N, T, D)
        y (np.ndarray): shape (N,)
        le (LabelEncoder): fitted label encoder
    """
    X, y, le = load_dlwf_npz(
        npz_path=npz_path,
        normalize=normalize,
        label_encoder=None,
    )

    if num_classes is not None:
        selected_class_ids = np.arange(min(num_classes, len(le.classes_)))
        mask = np.isin(y, selected_class_ids)
        X = X[mask]
        y = y[mask]

    if max_samples is not None and max_samples < len(X):
        rng = np.random.default_rng(random_state)
        indices = rng.choice(len(X), size=max_samples, replace=False)
        X = X[indices]
        y = y[indices]

    return X, y, le


# --------------------------------------------------
# Time-series augmentations for BYOL
# --------------------------------------------------


class TimeSeriesAugmenter:
    def __init__(
        self,
        jitter_scale=0.02,
        scaling_sigma=0.1,
        crop_ratio_min=0.8,
        crop_ratio_max=1.0,
        masking_ratio=0.05,
        time_flip=False,
    ):
        self.jitter_scale = jitter_scale
        self.scaling_sigma = scaling_sigma
        self.crop_ratio_min = crop_ratio_min
        self.crop_ratio_max = crop_ratio_max
        self.masking_ratio = masking_ratio
        self.time_flip = time_flip

    def _jitter(self, x: np.ndarray) -> np.ndarray:
        if self.jitter_scale <= 0:
            return x
        noise = np.random.normal(loc=0.0, scale=self.jitter_scale, size=x.shape)
        return x + noise.astype(np.float32)

    def _scaling(self, x: np.ndarray) -> np.ndarray:
        if self.scaling_sigma <= 0:
            return x
        scale = np.random.normal(loc=1.0, scale=self.scaling_sigma, size=(1, x.shape[1]))
        return x * scale.astype(np.float32)

    def _random_mask(self, x: np.ndarray) -> np.ndarray:
        if self.masking_ratio <= 0:
            return x
        x = x.copy()
        t = x.shape[0]
        n_mask = max(1, int(round(t * self.masking_ratio)))
        idx = np.random.choice(t, size=n_mask, replace=False)
        x[idx] = 0.0
        return x

    def _random_crop_and_resize(self, x: np.ndarray) -> np.ndarray:
        t, d = x.shape
        min_len = max(2, int(round(t * self.crop_ratio_min)))
        max_len = max(min_len, int(round(t * self.crop_ratio_max)))
        crop_len = np.random.randint(min_len, max_len + 1)
        start = np.random.randint(0, t - crop_len + 1)
        cropped = x[start:start + crop_len]

        if crop_len == t:
            return cropped

        old_idx = np.linspace(0, 1, num=crop_len)
        new_idx = np.linspace(0, 1, num=t)
        resized = np.zeros((t, d), dtype=np.float32)
        for ch in range(d):
            resized[:, ch] = np.interp(new_idx, old_idx, cropped[:, ch])
        return resized

    def _maybe_flip(self, x: np.ndarray) -> np.ndarray:
        if self.time_flip and np.random.rand() < 0.5:
            return x[::-1].copy()
        return x

    def __call__(self, x: np.ndarray) -> np.ndarray:
        out = x.astype(np.float32, copy=True)
        out = self._random_crop_and_resize(out)
        out = self._jitter(out)
        out = self._scaling(out)
        out = self._random_mask(out)
        out = self._maybe_flip(out)
        return out


class DLWFBYOLDataset(Dataset):
    def __init__(self, X, y, augmenter: TimeSeriesAugmenter):
        self.X = X
        self.y = y
        self.augmenter = augmenter

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = self.X[idx]
        y = self.y[idx]
        view1 = self.augmenter(x)
        view2 = self.augmenter(x)
        return (
            torch.tensor(view1, dtype=torch.float32),
            torch.tensor(view2, dtype=torch.float32),
            torch.tensor(y, dtype=torch.long),
        )


class DLWFClassificationDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# --------------------------------------------------
# Model
# --------------------------------------------------


class ResidualBlock1D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=5,
            stride=stride,
            padding=2,
            bias=False,
        )
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(
            out_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        if in_channels != out_channels or stride != 1:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        identity = self.shortcut(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.relu(out + identity)
        return out


class BYOL1DEncoder(nn.Module):
    def __init__(self, input_dims: int = 1, hidden_dims: int = 64, repr_dims: int = 128):
        super().__init__()
        c1 = hidden_dims
        c2 = hidden_dims * 2
        c3 = hidden_dims * 4

        self.stem = nn.Sequential(
            nn.Conv1d(input_dims, c1, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm1d(c1),
            nn.ReLU(inplace=True),
        )
        self.layer1 = ResidualBlock1D(c1, c1, stride=1)
        self.layer2 = ResidualBlock1D(c1, c2, stride=2)
        self.layer3 = ResidualBlock1D(c2, c3, stride=2)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.proj = nn.Linear(c3, repr_dims)

    def forward(self, x):
        x = x.transpose(1, 2)  # (B, T, D) -> (B, D, T)
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.pool(x).squeeze(-1)
        x = self.proj(x)
        return x


class MLPHead(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x):
        return self.net(x)


class BYOLNet(nn.Module):
    def __init__(self, input_dims: int, hidden_dims: int, repr_dims: int, proj_hidden_dim: int, proj_dims: int):
        super().__init__()
        self.encoder = BYOL1DEncoder(
            input_dims=input_dims,
            hidden_dims=hidden_dims,
            repr_dims=repr_dims,
        )
        self.projector = MLPHead(
            in_dim=repr_dims,
            hidden_dim=proj_hidden_dim,
            out_dim=proj_dims,
        )

    def forward(self, x):
        h = self.encoder(x)
        z = self.projector(h)
        return h, z


class BYOLModel(nn.Module):
    def __init__(
        self,
        input_dims: int,
        hidden_dims: int,
        repr_dims: int,
        proj_hidden_dim: int,
        proj_dims: int,
        pred_hidden_dim: int,
        ema_decay: float,
    ):
        super().__init__()
        self.online = BYOLNet(input_dims, hidden_dims, repr_dims, proj_hidden_dim, proj_dims)
        self.target = copy.deepcopy(self.online)
        for param in self.target.parameters():
            param.requires_grad = False

        self.predictor = MLPHead(
            in_dim=proj_dims,
            hidden_dim=pred_hidden_dim,
            out_dim=proj_dims,
        )
        self.ema_decay = ema_decay

    @torch.no_grad()
    def update_target(self):
        for target_param, online_param in zip(self.target.parameters(), self.online.parameters()):
            target_param.data.mul_(self.ema_decay).add_(online_param.data, alpha=1.0 - self.ema_decay)

    def forward_online(self, x):
        h, z = self.online(x)
        p = self.predictor(z)
        return h, z, p

    @torch.no_grad()
    def forward_target(self, x):
        h, z = self.target(x)
        return h, z.detach()

    def encode(self, x):
        return self.online.encoder(x)


# --------------------------------------------------
# Loss
# --------------------------------------------------


def byol_regression_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    pred = F.normalize(pred, dim=1)
    target = F.normalize(target, dim=1)
    return 2.0 - 2.0 * (pred * target).sum(dim=1).mean()


# --------------------------------------------------
# Training / encoding
# --------------------------------------------------


@dataclass
class EpochStats:
    epoch: int
    train_loss: float



def train_byol(model, train_loader, optimizer, scheduler, device, epochs, log_every=10, grad_clip=None):
    history = []
    model.train()

    for epoch in range(1, epochs + 1):
        running_loss = 0.0
        num_batches = 0

        for step, (x1, x2, _) in enumerate(train_loader, start=1):
            x1 = x1.to(device, non_blocking=True)
            x2 = x2.to(device, non_blocking=True)

            _, _, p1 = model.forward_online(x1)
            _, _, p2 = model.forward_online(x2)

            with torch.no_grad():
                _, z1_t = model.forward_target(x1)
                _, z2_t = model.forward_target(x2)

            loss = 0.5 * (byol_regression_loss(p1, z2_t) + byol_regression_loss(p2, z1_t))

            optimizer.zero_grad()
            loss.backward()
            if grad_clip is not None and grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            model.update_target()

            running_loss += float(loss.item())
            num_batches += 1

            if log_every > 0 and step % log_every == 0:
                print(f"Epoch {epoch:03d} | Step {step:04d}/{len(train_loader):04d} | Loss {loss.item():.6f}")

        if scheduler is not None:
            scheduler.step()

        avg_loss = running_loss / max(1, num_batches)
        history.append(EpochStats(epoch=epoch, train_loss=avg_loss))
        print(f"[Epoch {epoch:03d}] avg train loss = {avg_loss:.6f}")

    return history


@torch.no_grad()
def encode_dataset(model, loader, device):
    model.eval()
    reps = []
    labels = []
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        h = model.encode(x)
        reps.append(h.cpu().numpy())
        labels.append(y.numpy())
    model.train()
    return np.concatenate(reps, axis=0), np.concatenate(labels, axis=0)


# --------------------------------------------------
# Main
# --------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="DLWF + BYOL training")

    parser.add_argument("--data-path", type=str, default=DATA_PATH_DEFAULT)
    parser.add_argument("--save-dir", type=str, default="outputs_byol_dlwf")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--normalize", action="store_true")
    parser.add_argument("--gpu", type=int, default=0, help="GPU id, use -1 for CPU")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--random-state", type=int, default=42)

    # Training args aligned with the TS2Vec script style.
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--ema-decay", type=float, default=0.996)
    parser.add_argument("--grad-clip", type=float, default=1.0)

    # Encoder / projector / predictor sizes.
    parser.add_argument("--repr-dims", type=int, default=128)
    parser.add_argument("--hidden-dims", type=int, default=64)
    parser.add_argument("--proj-hidden-dims", type=int, default=256)
    parser.add_argument("--proj-dims", type=int, default=128)
    parser.add_argument("--pred-hidden-dims", type=int, default=256)

    # Demo controls similar to the TS2Vec script.
    parser.add_argument("--num-classes", type=int, default=10, help="Only keep the first K classes for a quick demo")
    parser.add_argument("--max-samples", type=int, default=500, help="Maximum number of total samples for demo")

    # BYOL augmentation controls.
    parser.add_argument("--jitter-scale", type=float, default=0.02)
    parser.add_argument("--scaling-sigma", type=float, default=0.1)
    parser.add_argument("--crop-ratio-min", type=float, default=0.8)
    parser.add_argument("--crop-ratio-max", type=float, default=1.0)
    parser.add_argument("--masking-ratio", type=float, default=0.05)
    parser.add_argument("--time-flip", action="store_true")

    # Evaluation classifier.
    parser.add_argument("--clf-max-iter", type=int, default=2000)

    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    set_seed(args.random_state)
    device = get_device(args.gpu)

    print("====================================")
    print("DLWF + BYOL Training")
    print("====================================")
    print("Data:", args.data_path)
    print("Save dir:", args.save_dir)
    print("Device:", device)
    print("Num classes:", args.num_classes)
    print("Max samples:", args.max_samples)

    X, y, le = prepare_dlwf_data(
        npz_path=args.data_path,
        normalize=args.normalize,
        num_classes=args.num_classes,
        max_samples=args.max_samples,
        random_state=args.random_state,
    )

    print("Dataset shape:", X.shape)
    print("Classes kept:", len(np.unique(y)))
    print("Original label space size:", len(le.classes_))

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=args.test_size,
        stratify=y,
        random_state=args.random_state,
    )

    print("Train:", X_train.shape)
    print("Test:", X_test.shape)

    augmenter = TimeSeriesAugmenter(
        jitter_scale=args.jitter_scale,
        scaling_sigma=args.scaling_sigma,
        crop_ratio_min=args.crop_ratio_min,
        crop_ratio_max=args.crop_ratio_max,
        masking_ratio=args.masking_ratio,
        time_flip=args.time_flip,
    )

    ssl_train_dataset = DLWFBYOLDataset(X_train, y_train, augmenter=augmenter)
    drop_last = len(ssl_train_dataset) >= args.batch_size
    ssl_train_loader = DataLoader(
        ssl_train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=drop_last,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    eval_train_loader = DataLoader(
        DLWFClassificationDataset(X_train, y_train),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )
    eval_test_loader = DataLoader(
        DLWFClassificationDataset(X_test, y_test),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    model = BYOLModel(
        input_dims=X_train.shape[-1],
        hidden_dims=args.hidden_dims,
        repr_dims=args.repr_dims,
        proj_hidden_dim=args.proj_hidden_dims,
        proj_dims=args.proj_dims,
        pred_hidden_dim=args.pred_hidden_dims,
        ema_decay=args.ema_decay,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, args.epochs))

    print("\nTraining BYOL encoder...")
    history = train_byol(
        model=model,
        train_loader=ssl_train_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        epochs=args.epochs,
        log_every=max(1, len(ssl_train_loader) // 5) if len(ssl_train_loader) > 0 else 1,
        grad_clip=args.grad_clip,
    )

    print("\nEncoding representations...")
    train_repr, train_eval_labels = encode_dataset(model, eval_train_loader, device)
    test_repr, test_eval_labels = encode_dataset(model, eval_test_loader, device)

    print("Representation shape:", train_repr.shape)

    scaler = StandardScaler()
    train_repr_std = scaler.fit_transform(train_repr)
    test_repr_std = scaler.transform(test_repr)

    clf = LogisticRegression(max_iter=args.clf_max_iter, n_jobs=-1)
    print("\nTraining linear classifier...")
    clf.fit(train_repr_std, train_eval_labels)

    y_pred = clf.predict(test_repr_std)
    acc = accuracy_score(test_eval_labels, y_pred)
    report = classification_report(test_eval_labels, y_pred, digits=4)

    print("\n======================")
    print("RESULTS")
    print("======================")
    print("Accuracy:", acc)
    print(report)

    torch.save(
        {
            "online_state_dict": model.online.state_dict(),
            "target_state_dict": model.target.state_dict(),
            "predictor_state_dict": model.predictor.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "args": vars(args),
        },
        os.path.join(args.save_dir, "byol_encoder.pt"),
    )

    np.save(os.path.join(args.save_dir, "train_repr.npy"), train_repr_std)
    np.save(os.path.join(args.save_dir, "test_repr.npy"), test_repr_std)
    np.save(os.path.join(args.save_dir, "y_test.npy"), test_eval_labels)
    np.save(os.path.join(args.save_dir, "y_pred.npy"), y_pred)

    with open(os.path.join(args.save_dir, "classification_report.txt"), "w", encoding="utf-8") as f:
        f.write(report)

    with open(os.path.join(args.save_dir, "label_mapping.txt"), "w", encoding="utf-8") as f:
        for i, label in enumerate(le.classes_):
            f.write(f"{i}\t{label}\n")

    save_json(
        os.path.join(args.save_dir, "metrics.json"),
        {
            "accuracy": float(acc),
            "epochs": args.epochs,
            "train_size": int(len(X_train)),
            "test_size": int(len(X_test)),
            "representation_dim": int(train_repr.shape[1]),
            "history": [stats.__dict__ for stats in history],
        },
    )

    save_json(os.path.join(args.save_dir, "config.json"), vars(args))

    print("\nOutputs saved to:", args.save_dir)


if __name__ == "__main__":
    main()
