# PMSP Torch
# Ian Dennis Miller, Brian Lam, Blair Armstrong

import logging
import torch
import torch.nn as nn
import torch.optim as optim

from .util import write_losses, make_folder


class PMSPTrainer:
    def __init__(self, network, dataloader):
        self.network = network
        self.dataloader = dataloader

    def train_one(self):
        avg_loss = 0

        for step_idx, (frequency, graphemes, phonemes) in enumerate(self.dataloader):

            # forward pass
            outputs = self.network(graphemes)

            # calculate loss
            loss = self.criterion(outputs, phonemes)
            loss = (loss * frequency.view(-1, 1)).mean()
            avg_loss += loss.item()

            # backprop
            loss.backward()

            # optimize
            self.optimizer.step()
            self.optimizer.zero_grad()
        
        # create record of loss per epoch
        avg_loss = avg_loss / len(self.dataloader)
        return(avg_loss)


    def train(self, update_interval=10, num_epochs=300, learning_rate=0.001):

        self.folder = make_folder()
        self.losses = []

        self.criterion = nn.BCELoss(reduction='none')
        self.optimizer = optim.Adam(self.network.parameters(), lr=learning_rate)

        for epoch in range(num_epochs):
            avg_loss = self.train_one()
            self.losses.append(avg_loss)

            msg = "[EPOCH {}] loss: {:.10f}".format(epoch+1, avg_loss)
            if epoch % update_interval == 0:
                logging.info(msg)
            else:
                logging.debug(msg)

        # write plot of loss over time
        write_losses(self.losses, self.folder)
