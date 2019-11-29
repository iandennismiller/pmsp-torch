# PMSP Torch
# CAP Lab

from .stimuli import PMSPStimuli

import torch
import torch.nn as nn
import torch.optim as optim

class Trainable(nn.Module):
    def __init__(self, learning_rate=0.001):
        super(Trainable, self).__init__()
        self.learning_rate = learning_rate
        self.stimuli = PMSPStimuli()

    def train(self, learning_rate=0.001):
        avg_loss = 0

        for i, data in enumerate(self.stimuli.train_loader):
            (frequency, graphemes, phonemes) = data
            freq = frequency.float().view(-1, 1)
            inputs = graphemes.float()
            targets = phonemes.float()
            
            # forward pass
            outputs = self(inputs)

            # calculate loss
            loss = self.criterion(outputs, targets)
            loss = (loss * freq).mean()
            avg_loss += loss.item()

            # backprop
            loss.backward()

            # optimize
            self.optimizer.step()
            self.optimizer.zero_grad()
        
        # create record of loss per epoch
        avg_loss = avg_loss / len(self.stimuli.train_loader)
        return(avg_loss)

class PMSPNet(Trainable):
    def __init__(self, learning_rate=0.001):
        super(PMSPNet, self).__init__(learning_rate=learning_rate)

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

class PMSPDoubleNet(Trainable):
    def __init__(self, learning_rate=0.001):
        super(PMSPDoubleNet, self).__init__(learning_rate=learning_rate)

        self.input_size = 105
        self.hidden1_size = 64
        self.hidden2_size = 32
        self.hidden3_size = 16
        self.hidden4_size = 32
        self.output_size = 61

        self.layer1 = nn.Linear(self.input_size, self.hidden1_size)
        self.layer2 = nn.Linear(self.hidden1_size, self.hidden2_size)
        self.layer3 = nn.Linear(self.hidden2_size, self.hidden3_size)
        self.layer4 = nn.Linear(self.hidden3_size, self.hidden4_size)
        self.layer5 = nn.Linear(self.hidden4_size, self.output_size)
        self.init_weights()

        self.criterion = nn.BCELoss(reduction='none')
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)

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
        x = torch.sigmoid(self.layer4(x))
        x = torch.sigmoid(self.layer5(x))
        return x
