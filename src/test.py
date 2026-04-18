import argparse
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from data import load_and_preprocess, CLASSES, INPUT_DIM, NUM_CLASSES, IMG_SIZE
from model import MLP


def confusion_matrix(y_true, y_pred, n=NUM_CLASSES):
    cm = np.zeros((n, n), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    return cm


def print_report(cm):
    max_w = max(len(c) for c in CLASSES)
    header = ' ' * (max_w + 3) + '  '.join(f"{c[:6]:>6}" for c in CLASSES)
    print('\nConfusion Matrix (row=true, col=pred):')
    print(header)
    print('-' * len(header))
    for i, name in enumerate(CLASSES):
        print(f"{name:<{max_w}} | " + '  '.join(f"{cm[i,j]:>6}" for j in range(NUM_CLASSES)))

    print(f"\n{'Class':<25} {'Precision':>10} {'Recall':>10} {'N':>6}")
    print('-' * 53)
    for i, name in enumerate(CLASSES):
        tp = cm[i, i]
        prec = tp / cm[:, i].sum() if cm[:, i].sum() > 0 else 0
        rec  = tp / cm[i, :].sum() if cm[i, :].sum() > 0 else 0
        print(f"{name:<25} {prec:>10.4f} {rec:>10.4f} {cm[i,:].sum():>6}")


def plot_cm(cm, out_dir):
    norm = cm.astype(float) / cm.sum(axis=1, keepdims=True).clip(min=1)
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(norm, cmap='Blues')
    plt.colorbar(im, ax=ax)
    ax.set_xticks(range(NUM_CLASSES)); ax.set_yticks(range(NUM_CLASSES))
    ax.set_xticklabels(CLASSES, rotation=45, ha='right', fontsize=8)
    ax.set_yticklabels(CLASSES, fontsize=8)
    thresh = norm.max() / 2
    for i in range(NUM_CLASSES):
        for j in range(NUM_CLASSES):
            ax.text(j, i, f'{norm[i,j]:.2f}', ha='center', va='center', fontsize=6,
                    color='white' if norm[i,j] > thresh else 'black')
    ax.set(ylabel='True', xlabel='Predicted', title='Confusion Matrix')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'confusion_matrix.png'), dpi=150, bbox_inches='tight')
    plt.close()


def plot_errors(X_test, y_test, preds, error_idx, mean, std, out_dir, n_show=16):
    n_show = min(n_show, len(error_idx))
    chosen = np.random.default_rng(0).choice(error_idx, n_show, replace=False)
    ncols = 4
    nrows = (n_show + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 3, nrows * 3))
    for ax, idx in zip(axes.flat, chosen):
        img = np.clip((X_test[idx] * std.flatten() + mean.flatten()).reshape(IMG_SIZE, IMG_SIZE, 3), 0, 1)
        ax.imshow(img); ax.axis('off')
        ax.set_title(f"true: {CLASSES[y_test[idx]]}\npred: {CLASSES[preds[idx]]}", fontsize=7, color='red')
    for ax in axes.flat[n_show:]:
        ax.axis('off')
    plt.suptitle('Error Analysis', fontsize=11)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'error_examples.png'), dpi=150, bbox_inches='tight')
    plt.close()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data_dir',     default='EuroSAT_RGB')
    p.add_argument('--weight_path',  default='best_model.npz')
    p.add_argument('--norm_stats',   default='outputs/norm_stats.npz')
    p.add_argument('--hidden_dim1',  type=int, default=512)
    p.add_argument('--hidden_dim2',  type=int, default=256)
    p.add_argument('--activation',   default='relu')
    p.add_argument('--output_dir',   default='outputs')
    p.add_argument('--seed',         type=int, default=42)
    args = p.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    data = load_and_preprocess(args.data_dir, seed=args.seed)
    X_test, y_test = data['X_test'], data['y_test']

    if os.path.exists(args.norm_stats):
        stats = np.load(args.norm_stats)
        mean, std = stats['mean'], stats['std']
    else:
        mean, std = data['mean'], data['std']

    model = MLP(INPUT_DIM, args.hidden_dim1, args.hidden_dim2, NUM_CLASSES, args.activation)
    model.load(args.weight_path)
    print(f"Loaded weights: {args.weight_path}")

    preds = np.concatenate([
        model.predict(X_test[s:s+512]) for s in range(0, len(X_test), 512)
    ])
    acc = (preds == y_test).mean()
    cm  = confusion_matrix(y_test, preds)

    print(f"\nTest Accuracy: {acc:.4f}  ({int(acc*len(y_test))}/{len(y_test)})")
    print_report(cm)

    plot_cm(cm, args.output_dir)
    plot_errors(X_test, y_test, preds, np.where(preds != y_test)[0], mean, std, args.output_dir)

    # first-layer weight visualization
    from train import plot_weights
    plot_weights(model.fc1.W, mean, std, args.output_dir)

    print(f"\nOutputs saved to: {args.output_dir}/")


if __name__ == '__main__':
    main()
