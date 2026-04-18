"""Numerical gradient check for backprop correctness."""

import numpy as np
from model import MLP


def numerical_grad(f, x, eps=1e-5):
    grad = np.zeros_like(x)
    for idx in np.ndindex(*x.shape):
        orig = x[idx]
        x[idx] = orig + eps; fp = f(x)
        x[idx] = orig - eps; fm = f(x)
        x[idx] = orig
        grad[idx] = (fp - fm) / (2 * eps)
    return grad


def check():
    np.random.seed(0)
    model = MLP(input_dim=30, hidden_dim1=16, hidden_dim2=8,
                num_classes=5, activation='relu', weight_decay=1e-3)
    X = np.random.randn(4, 30)
    y = np.random.randint(0, 5, 4)

    # analytic gradients
    _, dlogits = model.loss(X, y)
    model.backward(dlogits)
    analytic = {n: getattr(model, f'fc{i}').dW.copy()
                for i, n in [(1,'W1'),(2,'W2'),(3,'W3')]}

    # numerical gradients
    def loss_for(fc_attr, W):
        getattr(model, fc_attr).W = W.copy()
        l, _ = model.loss(X, y)
        return l

    numerical = {
        'W1': numerical_grad(lambda W: loss_for('fc1', W), model.fc1.W.copy()),
        'W2': numerical_grad(lambda W: loss_for('fc2', W), model.fc2.W.copy()),
        'W3': numerical_grad(lambda W: loss_for('fc3', W), model.fc3.W.copy()),
    }

    print(f"{'Param':<6}  {'Rel Error':>12}")
    print('-' * 22)
    for k in ['W1', 'W2', 'W3']:
        a, n = analytic[k], numerical[k]
        err = np.abs(a - n).max() / (np.abs(a).max() + np.abs(n).max() + 1e-12)
        print(f"  {k:<6}  {err:>12.2e}  {'PASS' if err < 1e-4 else 'FAIL'}")


if __name__ == '__main__':
    check()
