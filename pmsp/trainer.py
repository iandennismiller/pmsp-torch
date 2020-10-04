# PMSP Torch
# Ian Dennis Miller, Brian Lam, Blair Armstrong

import logging
import torch
import torch.nn as nn


class PMSPTrainer:
    def __init__(self, network):
        self.network = network
        self.epoch = 0

    def train_one(self, dataloader):
        avg_loss = 0

        for step_idx, (frequency, graphemes, phonemes) in enumerate(dataloader):
            assert step_idx is not None

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
        avg_loss = avg_loss / len(dataloader)
        return(avg_loss)


    def train(self, dataloader, num_epochs, optimizers, update_interval=10):
        self.losses = []
        self.criterion = nn.BCELoss(reduction='none')

        for epoch_iterator in range(num_epochs):
            # switch optimizer
            if self.epoch in optimizers:
                self.optimizer = optimizers[self.epoch]
                logging.info(f"switch to {self.optimizer}")

            # train one epoch and store the average loss
            avg_loss = self.train_one(dataloader)
            self.losses.append(avg_loss)

            msg = "[EPOCH {}] loss: {:.10f}".format(self.epoch, avg_loss)
            if epoch_iterator % update_interval == 0:
                logging.info(msg)
            else:
                logging.debug(msg)

            self.epoch += 1

        return self.losses
