"""Hyperparameter search (grid or random) over lr, hidden dims, weight_decay, activation."""

import os
import csv
import time
import itertools
import numpy as np

from data import load_and_preprocess, INPUT_DIM, NUM_CLASSES
from model import MLP, SGD
from train import train

DEFAULT_GRID = {
    'lr':           [0.05, 0.01, 0.001],
    'hidden_dim1':  [512, 1024, 2048],
    'hidden_dim2':  [256, 512, 1024],
    'weight_decay': [0.0, 5e-4, 1e-3],
    'activation':   ['relu', 'tanh'],
}


def _run(cfg, X_train, y_train, X_val, y_val, epochs, batch_size, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    tag = f"lr{cfg['lr']}_h{cfg['hidden_dim1']}-{cfg['hidden_dim2']}_wd{cfg['weight_decay']}_{cfg['activation']}"
    save_path = os.path.join(save_dir, f"model_{tag}.npz")

    # 已有权重文件则跳过，直接读取已有结果
    if os.path.exists(save_path):
        print(f"  [skip] {tag} (already done)")
        from train import evaluate
        model = MLP(INPUT_DIM, cfg['hidden_dim1'], cfg['hidden_dim2'],
                    NUM_CLASSES, cfg['activation'], cfg['weight_decay'])
        model.load(save_path)
        _, acc = evaluate(model, X_val, y_val)
        return {**cfg, 'best_val_acc': acc, 'elapsed': 0, 'weight_path': save_path}

    model = MLP(INPUT_DIM, cfg['hidden_dim1'], cfg['hidden_dim2'],
                NUM_CLASSES, cfg['activation'], cfg['weight_decay'])
    opt = SGD(cfg['lr'], decay_rate=0.5, decay_steps=10)

    t0 = time.time()
    hist = train(model, opt, X_train, y_train, X_val, y_val,
                 epochs=epochs, batch_size=batch_size, save_path=save_path)
    return {**cfg, 'best_val_acc': hist['val_acc'][int(np.argmax(hist['val_acc']))],
            'elapsed': round(time.time() - t0, 1), 'weight_path': save_path}


def grid_search(X_train, y_train, X_val, y_val, search_space=None,
                epochs=30, batch_size=256, save_dir='search_weights', result_csv='search_results.csv'):
    space = search_space or DEFAULT_GRID
    combos = list(itertools.product(*space.values()))
    print(f"Grid search: {len(combos)} configs × {epochs} epochs")
    results = [_run(dict(zip(space.keys(), c)), X_train, y_train, X_val, y_val,
                    epochs, batch_size, save_dir)
               for c in combos]
    results.sort(key=lambda r: r['best_val_acc'], reverse=True)
    _save_csv(results, result_csv)
    print(f"Top result: {results[0]}")
    return results


def random_search(X_train, y_train, X_val, y_val, n_trials=20, search_space=None,
                  epochs=30, batch_size=256, save_dir='search_weights',
                  result_csv='search_results.csv', seed=0):
    space = search_space or DEFAULT_GRID
    rng = np.random.default_rng(seed)
    configs = [{k: rng.choice(v).item() if hasattr(rng.choice(v), 'item') else rng.choice(v)
                for k, v in space.items()} for _ in range(n_trials)]
    print(f"Random search: {n_trials} trials × {epochs} epochs")
    results = [_run(cfg, X_train, y_train, X_val, y_val, epochs, batch_size, save_dir)
               for cfg in configs]
    results.sort(key=lambda r: r['best_val_acc'], reverse=True)
    _save_csv(results, result_csv)
    print(f"Top result: {results[0]}")
    return results


def _save_csv(results, path):
    with open(path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=results[0].keys())
        w.writeheader(); w.writerows(results)
    print(f"Results saved to {path}")


if __name__ == '__main__':
    from data import load_and_preprocess
    data = load_and_preprocess('EuroSAT_RGB')
    grid_search(data['X_train'], data['y_train'], data['X_val'], data['y_val'], epochs=30)
