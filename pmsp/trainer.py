# PMSP Torch
# Ian Dennis Miller, Brian Lam, Blair Armstrong

import logging
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .stimuli import PMSPStimuli
from .dataset import PMSPDataset
from .device_dataloader import DeviceDataLoader
from .util import write_losses


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

    def train_one(self, learning_rate=0.001):
        avg_loss = 0

        for step_idx, (frequency, graphemes, phonemes) in enumerate(self.dataloader):

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


    def train_all(self, learning_rate=0.001, update_interval=10, num_epochs=300):

        self.losses = []

        for epoch in range(num_epochs):
            avg_loss = self.model.train_one_epoch(learning_rate)
            self.losses.append(avg_loss)

            msg = "[EPOCH {}] loss: {:.10f}".format(epoch+1, avg_loss)
            if epoch % update_interval == 0:
                logging.info(msg)
            else:
                logging.debug(msg)

        # write plot of loss over time
        write_losses(self.losses, self.folder)
