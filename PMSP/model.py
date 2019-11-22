# PMSP Torch

from .stimuli import PMSPStimuli

import torch
import torch.nn as nn

class PlautNet(nn.Module):
    def __init__(self):
        super(PlautNet, self).__init__()
        self.layer1 = nn.Linear(105, 100)
        self.layer2 = nn.Linear(100, 61)
        self.init_weights()

    def init_weights(self):
        initrange = 0.1

        self.layer1.weight.data.uniform_(-initrange, initrange)
        self.layer1.bias.data.uniform_(-1.85, -1.85)
        # self.layer1.bias.data.zero_()
        
        self.layer2.weight.data.uniform_(-initrange, initrange)
        self.layer2.bias.data.uniform_(-1.85, -1.85)
        # .zero_()

    def forward(self, x):
        x = torch.sigmoid(self.layer1(x))
        x = torch.sigmoid(self.layer2(x))
        # x = self.layer2(x)
        return x
