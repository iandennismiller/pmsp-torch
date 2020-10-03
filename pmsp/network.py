# PMSP Torch
# Ian Dennis Miller, Brian Lam, Blair Armstrong

import logging
import torch
import torch.nn as nn


class PMSPNetwork(nn.Module):

    def __init__(self):

        super(PMSPNetwork, self).__init__()

        self.input_size = 105
        self.hidden_size = 100
        self.output_size = 61

        self.layer1 = nn.Linear(self.input_size, self.hidden_size)
        self.layer2 = nn.Linear(self.hidden_size, self.output_size)

        self.init_weights()
        self.init_cuda()


    def init_cuda(self):
        if torch.cuda.is_available():
            logging.info("using CUDA")
            self.cuda()
        else:
            logging.info("using CPU")


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

    def save(self, filename):
        torch.save(self.state_dict(), filename)

    def load(self, filename):
        self.load_state_dict(torch.load(filename))
        self.eval()
