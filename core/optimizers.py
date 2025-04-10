import numpy as np

class Optimizer:
    def step(self, layer):
        raise NotImplementedError

    def zero_grad(self):
        pass

class SGD(Optimizer):
    def __init__(self, lr=0.01):
        self.lr = lr

    def step(self, layer):
        for param in layer.parameters():
            param['value'] -= self.lr * param['grad']


class Momentum(Optimizer):
    def __init__(self, lr=0.01, momentum=0.9):
        self.lr = lr
        self.momentum = momentum
        self.velocity = {}

    def step(self, layer):
        for i, param in enumerate(layer.parameters()):
            key = id(param['value'])
            if key not in self.velocity:
                self.velocity[key] = np.zeros_like(param['value'])
            self.velocity[key] = self.momentum * self.velocity[key] - self.lr * param['grad']
            param['value'] += self.velocity[key]


class Adam(Optimizer):
    def __init__(self, lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.t = 0
        self.m = {}
        self.v = {}

    def step(self, layer):
        self.t += 1
        for i, param in enumerate(layer.parameters()):
            key = id(param['value'])
            if key not in self.m:
                self.m[key] = np.zeros_like(param['value'])
                self.v[key] = np.zeros_like(param['value'])

            self.m[key] = self.beta1 * self.m[key] + (1 - self.beta1) * param['grad']
            self.v[key] = self.beta2 * self.v[key] + (1 - self.beta2) * (param['grad'] ** 2)

            m_hat = self.m[key] / (1 - self.beta1 ** self.t)
            v_hat = self.v[key] / (1 - self.beta2 ** self.t)

            param['value'] -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)