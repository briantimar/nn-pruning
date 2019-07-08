import torch
import numpy as numpy

def lenet300(torch.nn.Module):

    def __init__(self):
        self.hidden1 = torch.nn.Linear(300)
        self.hidden2 = torch.nn.Linear(100)
        self.activation = torch.nn.Relu()

    def forward(x):
        x = x.view(-1, 28**2)
        for l in [self.hidden1, self.hidden2]:
            x = l(x)
            x = self.activation(x)
        return x