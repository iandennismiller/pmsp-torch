# PMSP Torch
# Ian Dennis Miller, Brian Lam, Blair Armstrong

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .stimuli import PMSPStimuli
from .dataset import PMSPDataset
from .device_dataloader import DeviceDataLoader


class PMSPTrainer(nn.Module):
    def __init__(self, dataset, learning_rate=0.001):

        super(PMSPTrainer, self).__init__()

        self.learning_rate = learning_rate
        self.dataset = dataset

        self.dataloader = DeviceDataLoader(DataLoader(
            dataset,
            batch_size=len(self.dataset),
            num_workers=0
        ))

    def train(self, learning_rate=0.001):
        avg_loss = 0

        for i, data in enumerate(self.dataloader):
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
        avg_loss = avg_loss / len(self.dataloader)
        return(avg_loss)
