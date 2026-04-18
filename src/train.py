import argparse
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from data import load_and_preprocess, INPUT_DIM, NUM_CLASSES
from model import MLP, SGD


def evaluate(model, X, y, batch_size=512):
    n = X.shape[0]
    total_loss, correct, n_batches = 0.0, 0, 0
    for s in range(0, n, batch_size):
        xb, yb = X[s:s+batch_size], y[s:s+batch_size]
        l, _ = model.loss(xb, yb)
        correct   += (model.predict(xb) == yb).sum()
        total_loss += l
        n_batches  += 1
    return total_loss / n_batches, correct / n


def train(model, optimizer, X_train, y_train, X_val, y_val,
          epochs, batch_size, save_path):
    history = {'train_loss': [], 'val_loss': [], 'val_acc': []}
    best_acc = 0.0

    for epoch in range(epochs):
        optimizer.update_lr(epoch)
        # shuffle
        perm = np.random.permutation(len(X_train))
        Xs, ys = X_train[perm], y_train[perm]
        epoch_loss, n_batches = 0.0, 0
        for s in range(0, len(Xs), batch_size):
            xb, yb = Xs[s:s+batch_size], ys[s:s+batch_size]
            l, dlogits = model.loss(xb, yb)
            model.backward(dlogits)
            optimizer.step(model.layers)
            epoch_loss += l
            n_batches  += 1

        val_loss, val_acc = evaluate(model, X_val, y_val)
        history['train_loss'].append(epoch_loss / n_batches)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        saved = ''
        if val_acc > best_acc:
            best_acc = val_acc
            model.save(save_path)
            saved = '  *'

        print(f"Epoch {epoch+1:3d}/{epochs}  lr={optimizer.lr:.5f}  "
              f"train_loss={epoch_loss/n_batches:.4f}  "
              f"val_loss={val_loss:.4f}  val_acc={val_acc:.4f}{saved}")

    print(f"\nBest val acc: {best_acc:.4f}  weights: {save_path}")
    return history


def plot_curves(history, out_dir):
    epochs = range(1, len(history['train_loss']) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(epochs, history['train_loss'], label='train')
    ax1.plot(epochs, history['val_loss'],   label='val')
    ax1.set(xlabel='Epoch', ylabel='Loss', title='Loss')
    ax1.legend(); ax1.grid(alpha=0.3)

    ax2.plot(epochs, history['val_acc'], color='seagreen')
    best_ep = int(np.argmax(history['val_acc'])) + 1
    ax2.axvline(best_ep, color='gray', linestyle='--', alpha=0.5)
    ax2.set(xlabel='Epoch', ylabel='Accuracy', title='Val Accuracy', ylim=(0, 1))
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'training_curves.png'), dpi=150)
    plt.close()


def plot_weights(W1, mean, std, out_dir):
    """Reshape first-layer weights to image patches for visualization."""
    n_show, ncols = 64, 8
    nrows = n_show // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 1.4, nrows * 1.4))
    for i, ax in enumerate(axes.flat):
        # show raw weights — inverse-normalization drowns structure under mean image
        w = W1[:, i].reshape(64, 64, 3)
        w = (w - w.min()) / (w.max() - w.min() + 1e-8)
        ax.imshow(np.clip(w, 0, 1)); ax.axis('off')
    plt.suptitle('Layer-1 Weights', fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'weights_layer1.png'), dpi=150, bbox_inches='tight')
    plt.close()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data_dir',     default='EuroSAT_RGB')
    p.add_argument('--hidden_dim1',  type=int,   default=512)
    p.add_argument('--hidden_dim2',  type=int,   default=256)
    p.add_argument('--activation',   default='relu')
    p.add_argument('--lr',           type=float, default=0.01)
    p.add_argument('--decay_rate',   type=float, default=0.5)
    p.add_argument('--decay_steps',  type=int,   default=10)
    p.add_argument('--weight_decay', type=float, default=1e-4)
    p.add_argument('--epochs',       type=int,   default=50)
    p.add_argument('--batch_size',   type=int,   default=256)
    p.add_argument('--save_path',    default='best_model.npz')
    p.add_argument('--output_dir',   default='outputs')
    p.add_argument('--seed',         type=int,   default=42)
    args = p.parse_args()

    np.random.seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    data = load_and_preprocess(args.data_dir, seed=args.seed)
    np.savez(os.path.join(args.output_dir, 'norm_stats.npz'),
             mean=data['mean'], std=data['std'])

    model = MLP(INPUT_DIM, args.hidden_dim1, args.hidden_dim2,
                NUM_CLASSES, args.activation, args.weight_decay)
    optimizer = SGD(args.lr, args.decay_rate, args.decay_steps)

    history = train(model, optimizer,
                    data['X_train'], data['y_train'],
                    data['X_val'],   data['y_val'],
                    args.epochs, args.batch_size, args.save_path)

    model.load(args.save_path)
    plot_curves(history, args.output_dir)
    plot_weights(model.fc1.W, data['mean'], data['std'], args.output_dir)


if __name__ == '__main__':
    main()
