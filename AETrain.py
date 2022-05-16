import DataManager
import argparse
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import utils

from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
from AutoEncoder import autoencoder
from Environment import environment

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--stock_code", nargs="+",
                        default= ["010140", "006280", "009830",
                                  "011170", "010060", "034220",
                                  "000810"])
    args = parser.parse_args()

path_list = []
for stock_code in args.stock_code:
    path = utils.Base_DIR + "/" + stock_code
    path_list.append(path)

train_data, test_data = DataManager.get_data_tensor(path_list,
                                                    train_date_start="20090101",
                                                    train_date_end="20180101",
                                                    test_date_start="20180102",
                                                    test_date_end=None)

train_data = train_data.swapaxes(1, 2)
test_data = test_data.swapaxes(1, 2)
train_data = train_data.reshape(-1, 6)
test_data = test_data.reshape(-1, 6)

train_data = train_data[:,:-1] #Price data 제외
test_data = test_data[:,:-1] #Price data 제외
state_dim = 5

x_train = torch.tensor(train_data).float()
y_train = torch.tensor(train_data).float()
x_valid = torch.tensor(test_data).float()
y_valid = torch.tensor(test_data).float()

dataset = TensorDataset(x_train, y_train)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

class Train_AE(nn.Module):
    def __init__(self, AutoEncoder:nn.Module, lr:float):
        super().__init__()
        self.AutoEncoder = AutoEncoder
        self.lr = lr
        self.optimizer = torch.optim.Adam(params=self.AutoEncoder.parameters(), lr=self.lr)
        self.loss = nn.SmoothL1Loss()
        self.losses = []

    def train(self, epochs):
        for epoch in range(epochs + 1):
            for batch_idx, samples in enumerate(dataloader):
                x_train, y_train = samples
                y_hat = self.AutoEncoder(x_train)
                loss = self.loss(y_hat, y_train)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                np.set_printoptions(precision=5, suppress=True)
                print(f"Epoch:{epoch}/{epochs} | Batch: {batch_idx+1}/{len(dataloader)} | Loss: {loss.item()}")
                self.losses.append(loss.item())
        plt.plot(self.losses)
        plt.show()

    def validate(self, x_test, y_test):
        y_hat = self.AutoEncoder(x_test)
        val_loss = self.loss(y_hat, y_test)
        np.set_printoptions(precision=5, suppress=True)
        print(f"Val Loss: {val_loss.item()}")

    def save_model(self, path_encoder, path_decoder):
        torch.save(self.AutoEncoder.encoder.state_dict(), path_encoder)
        torch.save(self.AutoEncoder.decoder.state_dict(), path_decoder)


train_ae = Train_AE(AutoEncoder=autoencoder(5), lr=1e-3)
train_ae.train(epochs=100)
train_ae.validate(x_valid, y_valid)
train_ae.save_model(utils.SAVE_DIR + "AutoEncoder/encoder.pth",
                    utils.SAVE_DIR + "AutoEncoder/decoder.pth")



