# PMSP Torch
# Ian Dennis Miller, Brian Lam, Blair Armstrong

import torch
import torch.nn as nn
import torch.optim as optim

from .trainer import PMSPTrainer

class PMSPNetwork(PMSPTrainer):

    def __init__(self, dataset, learning_rate=0.001):

        super(PMSPNetwork, self).__init__(dataset, learning_rate=learning_rate)

        self.input_size = 105
        self.hidden_size = 100
        self.output_size = 61

        self.layer1 = nn.Linear(self.input_size, self.hidden_size)
        self.layer2 = nn.Linear(self.hidden_size, self.output_size)
        self.init_weights()

        self.criterion = nn.BCELoss(reduction='none')
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)

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
