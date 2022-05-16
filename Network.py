import torch
import torch.nn as nn


class Actor(nn.Module):
    def __init__(self, K):
        super(Actor, self).__init__()
        self.K = K

        self.layer0 = nn.Linear(7, 128)
        self.layer1 = nn.Linear(128*K, 128)
        self.layer2 = nn.Linear(128 ,64)
        self.layer3 = nn.Linear(64, 3)
        self.hidden_act = nn.ReLU()

    def forward(self, s1_tensor, portfolio):
        """
        state = (s1_tensor, portfolio)
        s1_tensor: (batch, assets, features)
        """

        for k in range(s1_tensor.shape[1]):
            s2 = torch.cat([portfolio[:,0], portfolio[:,k+1]], dim=-1)
            s = torch.cat([s1_tensor[:,k,:], s2.view(-1, 2)], dim=-1)
            globals()[f"score{k+1}"] = self.hidden_act(self.layer0(s))

        for j in range(s1_tensor.shape[1]):
            x_list = list() if j == 0 else x_list
            x_list.append(globals()[f"score{j+1}"])

        x = torch.cat(x_list, dim=-1)
        x = self.layer1(x)
        x = self.hidden_act(x)
        x = self.layer2(x)
        x = self.hidden_act(x)
        x = self.layer3(x)
        # x = torch.tanh(x)
        return x


class Critic(nn.Module):
    def __init__(self, K):
        super(Critic, self).__init__()
        self.K = K

        # header
        self.layer0 = nn.Linear(7, 128)
        self.layer1 = nn.Linear(128*K + K, 128)
        self.layer2 = nn.Linear(128, 64)
        self.layer3 = nn.Linear(64, 1)
        self.hidden_act = nn.ReLU()

        nn.init.kaiming_normal_(self.layer1.weight, nonlinearity="relu")
        nn.init.kaiming_normal_(self.layer2.weight, nonlinearity="relu")
        nn.init.kaiming_normal_(self.layer3.weight, nonlinearity="relu")

    def forward(self, s1_tensor, portfolio, action):

        for k in range(s1_tensor.shape[1]):
            s2 = torch.cat([portfolio[:,0], portfolio[:,k+1]], dim=-1)
            s = torch.cat([s1_tensor[:,k,:], s2.view(-1, 2)], dim=-1)
            globals()[f"score{k+1}"] = self.hidden_act(self.layer0(s))

        for j in range(s1_tensor.shape[1]):
            x_list = list() if j == 0 else x_list
            x_list.append(globals()[f"score{j+1}"])

        x = torch.cat(x_list + [action], dim=-1)
        x = self.layer1(x)
        x = self.hidden_act(x)
        x = self.layer2(x)
        x = self.hidden_act(x)
        q = self.layer3(x)
        return q



if __name__ == "__main__":
    s1_tensor = torch.rand(size=(10, 3, 5))
    portfolio = torch.rand(size=(10, 4))
    action = torch.rand(size=(10, 3))

    actor = Actor(K=3)
    critic = Critic(K=3)
    print(actor(s1_tensor,portfolio).shape)
    print(critic(s1_tensor,portfolio,action).shape)