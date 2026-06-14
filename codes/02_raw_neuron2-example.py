import math
import random

def sigmoid(x):
    return 1 / (1 + math.exp(-x))

class Neuron:
    def __init__(self):
        # soruda verilen hatalı değerler
        # self.w1 = 0.2
        # self.bias = -3

        # bizim bulduğumuz olması gereken kaysayılar.
        self.w1 = 1.13
        self.bias = -10.56

    def forward(self, x1):
        z = x1 * self.w1 + self.bias
        return sigmoid(z)


if __name__ == "__main__":
    neuron = Neuron()
    for soru in range(16):
        predict = neuron.forward(soru)
        if predict < 0.5:
            print(f"{soru} soru ile %{predict * 100:.2f} olasılıkla KALDI.")
        else:
            print(f"{soru} soru ile %{predict * 100:.2f} olasılıkla GEÇTİ.")
