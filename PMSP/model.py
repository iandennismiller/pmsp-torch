# PMSP Torch
# CAP Lab

from .stimuli import PMSPStimuli

import torch
import torch.nn as nn

class PMSPNet(nn.Module):
    def __init__(self):
        super(PMSPNet, self).__init__()

        self.input_size = 105
        self.hidden_size = 100
        self.output_size = 61

        self.layer1 = nn.Linear(self.input_size, self.hidden_size)
        self.layer2 = nn.Linear(self.hidden_size, self.output_size)
        self.init_weights()

    def init_weights(self):
        initrange = 0.1

        self.layer1.weight.data.uniform_(-initrange, initrange)
        self.layer1.bias.data.uniform_(-1.85, -1.85)
        
        self.layer2.weight.data.uniform_(-initrange, initrange)
        self.layer2.bias.data.uniform_(-1.85, -1.85)

    def forward(self, x):
        x = torch.sigmoid(self.layer1(x))
        x = torch.sigmoid(self.layer2(x))
        return x

class PMSPDoubleNet(nn.Module):
    def __init__(self):
        super(PMSPDoubleNet, self).__init__()

        self.input_size = 105
        self.hidden1_size = 64
        self.hidden2_size = 32
        self.output_size = 61

        self.layer1 = nn.Linear(self.input_size, self.hidden1_size)
        self.layer2 = nn.Linear(self.hidden1_size, self.hidden2_size)
        self.layer3 = nn.Linear(self.hidden2_size, self.output_size)
        self.init_weights()

    def init_weights(self):
        initrange = 0.1

        self.layer1.weight.data.uniform_(-initrange, initrange)
        self.layer1.bias.data.uniform_(-1.85, -1.85)
        
        self.layer2.weight.data.uniform_(-initrange, initrange)
        self.layer2.bias.data.uniform_(-1.85, -1.85)

        self.layer3.weight.data.uniform_(-initrange, initrange)
        self.layer3.bias.data.uniform_(-1.85, -1.85)

    def forward(self, x):
        x = torch.sigmoid(self.layer1(x))
        x = torch.sigmoid(self.layer2(x))
        x = torch.sigmoid(self.layer3(x))
        return x
