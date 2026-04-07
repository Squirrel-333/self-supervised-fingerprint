import os
import json
import argparse
import math
import random
from dataclasses import asdict, dataclass

import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from dlwf_dataloader import load_dlwf_npz


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



def prepare_dlwf_data(
    npz_path,
    normalize=False,
    num_classes=None,
    max_samples=None,
    random_state=42,
):
    """
    Shared dataset preparation logic, aligned with the user's TS2Vec script.
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


class SequenceDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# --------------------------------------------------
# 1D MAE model
# --------------------------------------------------


class MLP(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, dropout: float = 0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x):
        return self.net(x)


class TransformerBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 4.0, dropout: float = 0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, int(dim * mlp_ratio), dim, dropout=dropout)

    def forward(self, x):
        x_norm = self.norm1(x)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm, need_weights=False)
        x = x + attn_out
        x = x + self.mlp(self.norm2(x))
        return x


class PatchEmbed1D(nn.Module):
    def __init__(self, in_chans: int, patch_size: int, embed_dim: int):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv1d(
            in_channels=in_chans,
            out_channels=embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )

    def forward(self, x):
        # x: (B, T, C) -> (B, C, T)
        x = x.transpose(1, 2)
        x = self.proj(x)          # (B, D, num_patches)
        x = x.transpose(1, 2)     # (B, num_patches, D)
        return x


class MAE1D(nn.Module):
    def __init__(
        self,
        seq_len: int,
        in_chans: int = 1,
        patch_size: int = 16,
        embed_dim: int = 128,
        depth: int = 6,
        num_heads: int = 4,
        decoder_embed_dim: int = 64,
        decoder_depth: int = 2,
        decoder_num_heads: int = 4,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
    ):
        super().__init__()

        if seq_len % patch_size != 0:
            raise ValueError(
                f"Sequence length ({seq_len}) must be divisible by patch_size ({patch_size})."
            )

        self.seq_len = seq_len
        self.in_chans = in_chans
        self.patch_size = patch_size
        self.num_patches = seq_len // patch_size
        self.patch_dim = patch_size * in_chans
        self.embed_dim = embed_dim

        self.patch_embed = PatchEmbed1D(in_chans, patch_size, embed_dim)
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, embed_dim))

        self.encoder_blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, mlp_ratio=mlp_ratio, dropout=dropout)
            for _ in range(depth)
        ])
        self.encoder_norm = nn.LayerNorm(embed_dim)

        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, decoder_embed_dim))
        self.decoder_blocks = nn.ModuleList([
            TransformerBlock(decoder_embed_dim, decoder_num_heads, mlp_ratio=mlp_ratio, dropout=dropout)
            for _ in range(decoder_depth)
        ])
        self.decoder_norm = nn.LayerNorm(decoder_embed_dim)
        self.decoder_pred = nn.Linear(decoder_embed_dim, self.patch_dim)

        self.initialize_weights()

    def initialize_weights(self):
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.decoder_pos_embed, std=0.02)
        nn.init.trunc_normal_(self.mask_token, std=0.02)

        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.Conv1d):
                nn.init.kaiming_normal_(module.weight, nonlinearity="linear")
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.LayerNorm):
                nn.init.constant_(module.bias, 0)
                nn.init.constant_(module.weight, 1.0)

    def patchify(self, x):
        # x: (B, T, C) -> (B, num_patches, patch_size * C)
        B, T, C = x.shape
        x = x.reshape(B, self.num_patches, self.patch_size, C)
        x = x.reshape(B, self.num_patches, self.patch_dim)
        return x

    def unpatchify(self, x):
        # x: (B, num_patches, patch_dim) -> (B, T, C)
        B = x.shape[0]
        x = x.reshape(B, self.num_patches, self.patch_size, self.in_chans)
        x = x.reshape(B, self.seq_len, self.in_chans)
        return x

    def random_masking(self, x, mask_ratio: float):
        """
        x: [B, L, D]
        returns visible tokens, binary mask, and restoration indices.
        mask: 0 means keep, 1 means remove.
        """
        B, L, D = x.shape
        len_keep = int(L * (1.0 - mask_ratio))
        len_keep = max(1, len_keep)

        noise = torch.rand(B, L, device=x.device)
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(
            x,
            dim=1,
            index=ids_keep.unsqueeze(-1).repeat(1, 1, D),
        )

        mask = torch.ones(B, L, device=x.device)
        mask[:, :len_keep] = 0
        mask = torch.gather(mask, dim=1, index=ids_restore)
        return x_masked, mask, ids_restore

    def forward_encoder(self, x, mask_ratio: float):
        x = self.patch_embed(x)
        x = x + self.pos_embed
        x_vis, mask, ids_restore = self.random_masking(x, mask_ratio)

        for blk in self.encoder_blocks:
            x_vis = blk(x_vis)
        x_vis = self.encoder_norm(x_vis)
        return x_vis, mask, ids_restore

    def forward_decoder(self, x_vis, ids_restore):
        x = self.decoder_embed(x_vis)
        B, L_vis, D = x.shape
        L = ids_restore.shape[1]

        mask_tokens = self.mask_token.repeat(B, L - L_vis, 1)
        x_full = torch.cat([x, mask_tokens], dim=1)
        x_full = torch.gather(
            x_full,
            dim=1,
            index=ids_restore.unsqueeze(-1).repeat(1, 1, D),
        )

        x_full = x_full + self.decoder_pos_embed
        for blk in self.decoder_blocks:
            x_full = blk(x_full)
        x_full = self.decoder_norm(x_full)
        pred = self.decoder_pred(x_full)
        return pred

    def forward_loss(self, x, pred, mask):
        target = self.patchify(x)
        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)
        loss = (loss * mask).sum() / mask.sum().clamp_min(1.0)
        return loss

    def forward(self, x, mask_ratio: float = 0.75):
        latent, mask, ids_restore = self.forward_encoder(x, mask_ratio)
        pred = self.forward_decoder(latent, ids_restore)
        loss = self.forward_loss(x, pred, mask)
        return loss, pred, mask

    @torch.no_grad()
    def encode(self, x):
        # representation for downstream linear probing
        tokens = self.patch_embed(x)
        tokens = tokens + self.pos_embed
        for blk in self.encoder_blocks:
            tokens = blk(tokens)
        tokens = self.encoder_norm(tokens)
        reprs = tokens.mean(dim=1)
        return reprs


# --------------------------------------------------
# Training / evaluation
# --------------------------------------------------


def train_one_epoch(model, loader, optimizer, device, mask_ratio, grad_clip=None):
    model.train()
    total_loss = 0.0
    total_count = 0

    for x, _ in loader:
        x = x.to(device)
        optimizer.zero_grad()
        loss, _, _ = model(x, mask_ratio=mask_ratio)
        loss.backward()
        if grad_clip is not None and grad_clip > 0:
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        batch_size = x.size(0)
        total_loss += loss.item() * batch_size
        total_count += batch_size

    return total_loss / max(total_count, 1)


@torch.no_grad()
def extract_representations(model, X, batch_size, device):
    model.eval()
    dataset = torch.tensor(X, dtype=torch.float32)
    outputs = []
    for start in range(0, len(dataset), batch_size):
        batch = dataset[start:start + batch_size].to(device)
        reprs = model.encode(batch)
        outputs.append(reprs.cpu().numpy())
    return np.concatenate(outputs, axis=0)


@dataclass
class Metrics:
    mae_train_loss_last_epoch: float
    linear_probe_accuracy: float
    train_size: int
    test_size: int
    seq_len: int
    num_classes_kept: int


# --------------------------------------------------
# Main
# --------------------------------------------------


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data-path", type=str, default=DATA_PATH_DEFAULT)
    parser.add_argument("--save-dir", type=str, default="outputs_mae_dlwf_demo")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--normalize", action="store_true")
    parser.add_argument("--gpu", type=int, default=0, help="GPU id, use -1 for CPU")

    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=1.0)

    parser.add_argument("--patch-size", type=int, default=25)
    parser.add_argument("--embed-dim", type=int, default=128)
    parser.add_argument("--depth", type=int, default=6)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--decoder-embed-dim", type=int, default=64)
    parser.add_argument("--decoder-depth", type=int, default=2)
    parser.add_argument("--decoder-num-heads", type=int, default=4)
    parser.add_argument("--mlp-ratio", type=float, default=4.0)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--mask-ratio", type=float, default=0.75)

    parser.add_argument(
        "--num-classes",
        type=int,
        default=2,
        help="Only keep the first K classes for a quick demo",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=100,
        help="Maximum number of total samples for demo",
    )
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)

    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    set_seed(args.random_state)

    print("====================================")
    print("DLWF + MAE Demo Training")
    print("====================================")
    print("Data:", args.data_path)
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

    device = torch.device(
        "cpu" if args.gpu < 0 or not torch.cuda.is_available() else f"cuda:{args.gpu}"
    )
    print("Device:", device)

    seq_len = X_train.shape[1]
    in_chans = X_train.shape[2]
    if seq_len % args.patch_size != 0:
        raise ValueError(
            f"seq_len={seq_len} is not divisible by patch_size={args.patch_size}. "
            "Please set --patch-size to a divisor of the sequence length."
        )

    train_loader = DataLoader(
        SequenceDataset(X_train, y_train),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        drop_last=False,
    )

    model = MAE1D(
        seq_len=seq_len,
        in_chans=in_chans,
        patch_size=args.patch_size,
        embed_dim=args.embed_dim,
        depth=args.depth,
        num_heads=args.num_heads,
        decoder_embed_dim=args.decoder_embed_dim,
        decoder_depth=args.decoder_depth,
        decoder_num_heads=args.decoder_num_heads,
        mlp_ratio=args.mlp_ratio,
        dropout=args.dropout,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    loss_history = []
    print("\nTraining MAE...")
    for epoch in range(args.epochs):
        train_loss = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            device=device,
            mask_ratio=args.mask_ratio,
            grad_clip=args.grad_clip,
        )
        loss_history.append(train_loss)
        print(f"Epoch {epoch + 1:03d}/{args.epochs:03d} | train_loss={train_loss:.6f}")

    print("\nEncoding representations...")
    train_repr = extract_representations(model, X_train, args.batch_size, device)
    test_repr = extract_representations(model, X_test, args.batch_size, device)
    print("Representation shape:", train_repr.shape)

    scaler = StandardScaler()
    train_repr_std = scaler.fit_transform(train_repr)
    test_repr_std = scaler.transform(test_repr)

    clf = LogisticRegression(max_iter=2000, n_jobs=-1)
    print("\nTraining classifier...")
    clf.fit(train_repr_std, y_train)

    y_pred = clf.predict(test_repr_std)
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, digits=4)

    print("\n======================")
    print("RESULTS")
    print("======================")
    print("Accuracy:", acc)
    print(report)

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "args": vars(args),
            "seq_len": seq_len,
            "in_chans": in_chans,
            "label_classes": le.classes_.tolist(),
        },
        os.path.join(args.save_dir, "mae_encoder.pt"),
    )

    np.save(os.path.join(args.save_dir, "train_repr.npy"), train_repr_std)
    np.save(os.path.join(args.save_dir, "test_repr.npy"), test_repr_std)
    np.save(os.path.join(args.save_dir, "y_test.npy"), y_test)
    np.save(os.path.join(args.save_dir, "y_pred.npy"), y_pred)
    np.save(os.path.join(args.save_dir, "train_loss_history.npy"), np.array(loss_history, dtype=np.float32))

    with open(os.path.join(args.save_dir, "classification_report.txt"), "w", encoding="utf-8") as f:
        f.write(report)

    with open(os.path.join(args.save_dir, "label_mapping.txt"), "w", encoding="utf-8") as f:
        for i, label in enumerate(le.classes_):
            f.write(f"{i}\t{label}\n")

    metrics = Metrics(
        mae_train_loss_last_epoch=float(loss_history[-1]),
        linear_probe_accuracy=float(acc),
        train_size=int(len(X_train)),
        test_size=int(len(X_test)),
        seq_len=int(seq_len),
        num_classes_kept=int(len(np.unique(y))),
    )
    with open(os.path.join(args.save_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(asdict(metrics), f, indent=2)

    with open(os.path.join(args.save_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2)

    print("\nOutputs saved to:", args.save_dir)


if __name__ == "__main__":
    main()
