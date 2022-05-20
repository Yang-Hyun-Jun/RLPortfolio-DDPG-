import numpy as np

Base_DIR = "/Users/mac/Desktop/OHLCV_data/ALL_OHLCV"
SAVE_DIR = "/Users/mac/Desktop/RLPortfolio/DDPGPortfolio"

def sigmoid(x):
    x = np.clip(x, -10, 10)
    return 1. / (1. + np.exp(-x))

def exp(x):
    return np.exp(x)

if __name__ == "__main__":
    import torch
    a = torch.rand(size=(1, 3))
    print(torch.softmax(a, dim=1))