import math
import random


def sigmoid(x):
    return 1 / (1 + math.exp(-x))


class Neuron:
    def __init__(self):
        self.w1 = random.uniform(-1, 1)
        self.w2 = random.uniform(-1, 1)
        self.w3 = random.uniform(-1, 1)
        self.bias = random.uniform(-1, 1)

    def forward(self, x1, x2, x3):
        z = x1 * self.w1 + x2 * self.w2 + x3 * self.w3 + self.bias
        return sigmoid(z)


if __name__ == "__main__":
    neuron = Neuron()
    rooms = 3
    area = 150
    age = 2
    predict = neuron.forward(rooms, area, age)
    print(f"rooms w1 = {neuron.w1:.4f}")
    print(f"area  w2 = {neuron.w2:.4f}")
    print(f"age   w3 = {neuron.w3:.4f}")
    print(f"bias     = {neuron.bias:.4f}")
    print(f"Tahmin : {predict}\n")
