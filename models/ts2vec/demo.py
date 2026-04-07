import os
import sys
import argparse
import numpy as np

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# --------------------------------------------------
# Path setup
# --------------------------------------------------

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

TS2VEC_PATH = os.path.join(PROJECT_ROOT, "models", "ts2vec")
DATA_PATH_DEFAULT = os.path.join(
    PROJECT_ROOT,
    "data",
    "closed_world",
    "tor_100w_2500tr.npz"
)

sys.path.append(TS2VEC_PATH)

from ts2vec import TS2Vec


# --------------------------------------------------
# Dataset loader
# --------------------------------------------------

def load_dlwf_dataset(
    npz_path,
    normalize=False,
    num_classes=None,
    max_samples=None,
    random_state=42
):
    data_obj = np.load(npz_path, allow_pickle=True)

    if "data" not in data_obj.files or "labels" not in data_obj.files:
        raise ValueError(f"Expected keys ['data', 'labels'], got {data_obj.files}")

    X = data_obj["data"]       # shape (N, L)
    y_raw = data_obj["labels"]

    if X.ndim != 2:
        raise ValueError(f"Expected shape (N, L), got {X.shape}")

    # Optionally keep only a subset of classes
    if num_classes is not None:
        unique_labels = np.unique(y_raw)
        selected_labels = unique_labels[:num_classes]
        mask = np.isin(y_raw, selected_labels)
        X = X[mask]
        y_raw = y_raw[mask]

    # Optionally subsample the dataset
    if max_samples is not None and max_samples < len(X):
        rng = np.random.default_rng(random_state)
        indices = rng.choice(len(X), size=max_samples, replace=False)
        X = X[indices]
        y_raw = y_raw[indices]

    # Encode labels
    le = LabelEncoder()
    y = le.fit_transform(y_raw)

    # Convert to TS2Vec format (N, T, D)
    X = X.astype(np.float32)
    X = X[..., np.newaxis]

    if normalize:
        mean = X.mean()
        std = X.std()
        if std > 0:
            X = (X - mean) / std

    return X, y, le


# --------------------------------------------------
# Main training pipeline
# --------------------------------------------------

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data-path", type=str, default=DATA_PATH_DEFAULT)
    parser.add_argument("--save-dir", type=str, default="outputs_ts2vec_dlwf_demo")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--normalize", action="store_true")
    parser.add_argument("--gpu", type=int, default=0, help="GPU id, use -1 for CPU")

    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)

    parser.add_argument("--repr-dims", type=int, default=128)
    parser.add_argument("--hidden-dims", type=int, default=32)
    parser.add_argument("--depth", type=int, default=6)

    # Demo controls
    parser.add_argument("--num-classes", type=int, default=5,
                        help="Only keep the first K classes for a quick demo")
    parser.add_argument("--max-samples", type=int, default=10,
                        help="Maximum number of total samples for demo")
    parser.add_argument("--random-state", type=int, default=42)

    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    print("====================================")
    print("DLWF + TS2Vec Demo Training")
    print("====================================")
    print("Data:", args.data_path)
    print("Num classes:", args.num_classes)
    print("Max samples:", args.max_samples)

    # --------------------------------------------------
    # Load dataset
    # --------------------------------------------------

    X, y, le = load_dlwf_dataset(
        args.data_path,
        normalize=args.normalize,
        num_classes=args.num_classes,
        max_samples=args.max_samples,
        random_state=args.random_state
    )

    print("Dataset shape:", X.shape)
    print("Classes:", len(le.classes_))

    # --------------------------------------------------
    # Train/Test split
    # --------------------------------------------------

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=args.test_size,
        stratify=y,
        random_state=args.random_state
    )

    print("Train:", X_train.shape)
    print("Test:", X_test.shape)

    # --------------------------------------------------
    # Device
    # --------------------------------------------------

    device = "cpu" if args.gpu < 0 else args.gpu

    # --------------------------------------------------
    # Build TS2Vec model
    # --------------------------------------------------

    model = TS2Vec(
        input_dims=X_train.shape[-1],
        output_dims=args.repr_dims,
        hidden_dims=args.hidden_dims,
        depth=args.depth,
        device=device,
        lr=args.lr,
        batch_size=args.batch_size,
        max_train_length=5000
    )

    # --------------------------------------------------
    # Train encoder
    # --------------------------------------------------

    print("\nTraining TS2Vec...")
    model.fit(
        X_train,
        n_epochs=args.epochs,
        verbose=True
    )

    # --------------------------------------------------
    # Encode representations
    # --------------------------------------------------

    print("\nEncoding representations...")

    train_repr = model.encode(
        X_train,
        encoding_window="full_series"
    )

    test_repr = model.encode(
        X_test,
        encoding_window="full_series"
    )

    print("Representation shape:", train_repr.shape)

    # --------------------------------------------------
    # Standardize features
    # --------------------------------------------------

    scaler = StandardScaler()
    train_repr = scaler.fit_transform(train_repr)
    test_repr = scaler.transform(test_repr)

    # --------------------------------------------------
    # Train classifier
    # --------------------------------------------------

    clf = LogisticRegression(
        max_iter=2000,
        n_jobs=-1
    )

    print("\nTraining classifier...")
    clf.fit(train_repr, y_train)

    # --------------------------------------------------
    # Evaluate
    # --------------------------------------------------

    y_pred = clf.predict(test_repr)
    acc = accuracy_score(y_test, y_pred)

    print("\n======================")
    print("RESULTS")
    print("======================")
    print("Accuracy:", acc)
    print(classification_report(y_test, y_pred, digits=4))

    # --------------------------------------------------
    # Save results
    # --------------------------------------------------

    np.save(os.path.join(args.save_dir, "train_repr.npy"), train_repr)
    np.save(os.path.join(args.save_dir, "test_repr.npy"), test_repr)
    np.save(os.path.join(args.save_dir, "y_pred.npy"), y_pred)

    with open(os.path.join(args.save_dir, "label_mapping.txt"), "w", encoding="utf-8") as f:
        for i, label in enumerate(le.classes_):
            f.write(f"{i}\t{label}\n")

    print("\nOutputs saved to:", args.save_dir)


if __name__ == "__main__":
    main()