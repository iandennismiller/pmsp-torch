# PMSP Torch
# CAP Lab

from .model import PMSPNet
from .stimuli import PMSPStimuli
from .util import make_folder, write_losses

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

class Simulator:
    def __init__(self, batch_size=None, num_workers=None):
        torch.manual_seed(1)

        self.folder = make_folder()
        self.model = PMSPNet()
        if torch.cuda.is_available():
            print("using CUDA")
            self.model.cuda()
        else:
            print("using CPU")

        self.dataset = PMSPStimuli().dataset

        if not batch_size:
            batch_size = int(len(self.dataset)/30)

        if not num_workers:
            num_workers = 0

        self.train_loader = DataLoader(
            self.dataset,
            batch_size=batch_size,
            num_workers=num_workers
        )

    def train(self, learning_rate=0.001, num_epochs=300):
        criterion = nn.BCELoss(reduction='none')
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        losses = []

        for epoch in range(num_epochs):
            avg_loss = 0
            for i, data in enumerate(self.train_loader):
                freq = data["frequency"].float().view(-1, 1)
                inputs = data["graphemes"].float()
                targets = data["phonemes"].float()
                
                # forward pass
                outputs = self.model(inputs)

                # calculate loss
                loss = criterion(outputs, targets)
                loss = (loss * freq).mean()
                avg_loss += loss.item()

                # backprop
                loss.backward()

                # optimize
                optimizer.step()
                optimizer.zero_grad()
            
            # create record of loss per epoch
            avg_loss = avg_loss / len(self.train_loader)
            losses.append(avg_loss)
            print("[EPOCH {}] loss: {:.10f}".format(epoch+1, avg_loss))

        # write plot of loss over time
        write_losses(losses, self.folder)
