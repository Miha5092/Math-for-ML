import torch

from torch import nn

class SigmoidNeuralNetwork(nn.Module):
    def __init__(self, input_size: int, layer_size: int = 64, layer_no: int = 1):
        super().__init__()

        layers = [nn.Linear(input_size, layer_size), nn.ReLU(),]
        
        for _ in range(layer_no):
            layers.append(nn.Linear(layer_size, layer_size))
            layers.append(nn.ReLU())

        self.layers = nn.Sequential(*layers, nn.Linear(layer_size, 1))

        self.layers.apply(self._init_weights)

    def forward(self, x):
        return self.layers(x).view(-1)
    
    def clasiffy(self, x):
        return torch.round(nn.Sigmoid()(self.forward(x))).view(-1).long()
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.01)

class LogisticRegressionNet(nn.Module):
    def __init__(self, input_size: int):
        super().__init__()

        self.layers = nn.Linear(input_size, 1)

    def forward(self, x):
        return self.layers(x).view(-1)
    
    def clasiffy(self, x):
        return torch.round(nn.Sigmoid()(self.forward(x))).view(-1).long()