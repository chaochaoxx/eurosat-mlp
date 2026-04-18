import os
import numpy as np
from PIL import Image

CLASSES = [
    'AnnualCrop', 'Forest', 'HerbaceousVegetation', 'Highway',
    'Industrial', 'Pasture', 'PermanentCrop', 'Residential',
    'River', 'SeaLake',
]
NUM_CLASSES = 10
IMG_SIZE = 64
INPUT_DIM = IMG_SIZE * IMG_SIZE * 3  # 12288


def load_and_preprocess(data_dir, train_ratio=0.70, val_ratio=0.15, seed=42):
    X, y = [], []
    for cls_idx, cls_name in enumerate(CLASSES):
        cls_dir = os.path.join(data_dir, cls_name)
        for fname in sorted(os.listdir(cls_dir)):
            if not fname.lower().endswith(('.jpg', '.jpeg', '.png', '.tif', '.tiff')):
                continue
            img = Image.open(os.path.join(cls_dir, fname)).convert('RGB')
            X.append(np.array(img, dtype=np.float32).flatten())
            y.append(cls_idx)

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.int64)
    print(f"Loaded {len(X)} images, {NUM_CLASSES} classes")

    rng = np.random.default_rng(seed)
    tr_idx, val_idx, te_idx = [], [], []
    for c in range(NUM_CLASSES):
        idx = rng.permutation(np.where(y == c)[0])
        n = len(idx)
        n_tr = int(n * train_ratio)
        n_val = int(n * val_ratio)
        tr_idx.append(idx[:n_tr])
        val_idx.append(idx[n_tr:n_tr + n_val])
        te_idx.append(idx[n_tr + n_val:])

    tr_idx  = rng.permutation(np.concatenate(tr_idx))
    val_idx = rng.permutation(np.concatenate(val_idx))
    te_idx  = rng.permutation(np.concatenate(te_idx))
    print(f"Split: train={len(tr_idx)}, val={len(val_idx)}, test={len(te_idx)}")

    X_train, X_val, X_test = X[tr_idx] / 255.0, X[val_idx] / 255.0, X[te_idx] / 255.0
    mean = X_train.mean(axis=0, keepdims=True)
    std  = X_train.std(axis=0,  keepdims=True) + 1e-8

    return {
        'X_train': (X_train - mean) / std, 'y_train': y[tr_idx],
        'X_val':   (X_val   - mean) / std, 'y_val':   y[val_idx],
        'X_test':  (X_test  - mean) / std, 'y_test':  y[te_idx],
        'mean': mean, 'std': std,
    }
