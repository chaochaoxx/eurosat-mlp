import numpy as np
from data import NUM_CLASSES, INPUT_DIM


# ---------------------------------------------------------------------------
# Activations
# ---------------------------------------------------------------------------

class ReLU:
    def forward(self, x):
        self._x = x
        return np.maximum(0.0, x)

    def backward(self, dout):
        return dout * (self._x > 0)


class Sigmoid:
    def forward(self, x):
        # numerically stable split
        out = np.where(x >= 0, 1 / (1 + np.exp(-x)), np.exp(x) / (1 + np.exp(x)))
        self._out = out
        return out

    def backward(self, dout):
        return dout * self._out * (1 - self._out)


class Tanh:
    def forward(self, x):
        self._out = np.tanh(x)
        return self._out

    def backward(self, dout):
        return dout * (1 - self._out ** 2)


ACTIVATIONS = {'relu': ReLU, 'sigmoid': Sigmoid, 'tanh': Tanh}


# ---------------------------------------------------------------------------
# Linear layer
# ---------------------------------------------------------------------------

class Linear:
    def __init__(self, in_dim, out_dim, init='he'):
        scale = np.sqrt(2.0 / in_dim) if init == 'he' else np.sqrt(1.0 / in_dim)
        self.W  = np.random.randn(in_dim, out_dim).astype(np.float64) * scale
        self.b  = np.zeros(out_dim, dtype=np.float64)
        self.dW = None
        self.db = None

    def forward(self, x):
        self._x = x
        return x @ self.W + self.b

    def backward(self, dout):
        # dout is un-normalized (p - y); /N happens here to match mean CE loss
        n = self._x.shape[0]
        self.dW = self._x.T @ dout / n
        self.db = dout.mean(axis=0)
        return dout @ self.W.T


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------

def softmax(x):
    e = np.exp(x - x.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


def ce_loss(logits, labels):
    """Cross-entropy loss. Returns (scalar loss, dlogits un-normalized)."""
    n = logits.shape[0]
    probs = softmax(logits)
    loss = -np.log(np.clip(probs[np.arange(n), labels], 1e-12, 1.0)).mean()
    dlogits = probs.copy()
    dlogits[np.arange(n), labels] -= 1.0
    return loss, dlogits


# ---------------------------------------------------------------------------
# Optimizer
# ---------------------------------------------------------------------------

class SGD:
    def __init__(self, lr=0.01, decay_rate=1.0, decay_steps=10):
        self.lr0         = lr
        self.lr          = lr
        self.decay_rate  = decay_rate
        self.decay_steps = decay_steps

    def update_lr(self, epoch):
        self.lr = self.lr0 * (self.decay_rate ** (epoch // self.decay_steps))

    def step(self, layers):
        for layer in layers:
            if layer.dW is not None:
                layer.W -= self.lr * layer.dW
                layer.b -= self.lr * layer.db


# ---------------------------------------------------------------------------
# Three-layer MLP
# ---------------------------------------------------------------------------

class MLP:
    """
    Input → Linear → Act → Linear → Act → Linear → (softmax + CE)
    Three weight matrices = "three-layer" MLP.
    """
    def __init__(self, input_dim=INPUT_DIM, hidden_dim1=512, hidden_dim2=256,
                 num_classes=NUM_CLASSES, activation='relu', weight_decay=0.0):
        self.weight_decay = weight_decay
        init = 'he' if activation == 'relu' else 'xavier'
        self.fc1 = Linear(input_dim,   hidden_dim1, init)
        self.fc2 = Linear(hidden_dim1, hidden_dim2, init)
        self.fc3 = Linear(hidden_dim2, num_classes, 'xavier')
        self.layers = [self.fc1, self.fc2, self.fc3]
        act_cls = ACTIVATIONS[activation]
        self.act1 = act_cls()
        self.act2 = act_cls()

    def forward(self, x):
        return self.fc3.forward(self.act2.forward(self.fc2.forward(
               self.act1.forward(self.fc1.forward(x)))))

    def loss(self, x, labels):
        logits = self.forward(x)
        l, dlogits = ce_loss(logits, labels)
        reg = sum(0.5 * self.weight_decay * np.sum(fc.W ** 2) for fc in self.layers)
        return l + reg, dlogits

    def backward(self, dlogits):
        d = self.fc3.backward(dlogits)
        d = self.fc2.backward(self.act2.backward(d))
        self.fc1.backward(self.act1.backward(d))
        if self.weight_decay > 0:
            for fc in self.layers:
                fc.dW += self.weight_decay * fc.W

    def predict(self, x):
        return np.argmax(self.forward(x), axis=1)

    def save(self, path):
        np.savez(path, W1=self.fc1.W, b1=self.fc1.b,
                       W2=self.fc2.W, b2=self.fc2.b,
                       W3=self.fc3.W, b3=self.fc3.b)

    def load(self, path):
        d = np.load(path)
        for i, fc in enumerate([self.fc1, self.fc2, self.fc3], 1):
            fc.W, fc.b = d[f'W{i}'], d[f'b{i}']
