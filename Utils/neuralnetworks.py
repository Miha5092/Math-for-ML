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

    def forward(self, x):
        return self.layers(x).view(-1)
    
    def clasiffy(self, x):
        return torch.round(nn.Sigmoid()(self.forward(x))).view(-1).long()
