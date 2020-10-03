# PMSP Torch
# Ian Dennis Miller, Brian Lam, Blair Armstrong

import logging
import torch
import torch.nn as nn

from .util import write_figure, make_folder


class PMSPTrainer:
    def __init__(self, network, dataloader):
        self.network = network
        self.dataloader = dataloader

    def train_one(self):
        avg_loss = 0

        for step_idx, (frequency, graphemes, phonemes) in enumerate(self.dataloader):
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
        avg_loss = avg_loss / len(self.dataloader)
        return(avg_loss)


    def train(self, num_epochs, optimizers, update_interval=10):

        self.folder = make_folder()
        self.losses = []

        self.criterion = nn.BCELoss(reduction='none')

        for epoch in range(num_epochs):
            # switch optimizer
            if epoch in optimizers:
                self.optimizer = optimizers[epoch]
                logging.info(f"switch to {self.optimizer}")

            # train one epoch and store the average loss
            avg_loss = self.train_one()
            self.losses.append(avg_loss)

            msg = "[EPOCH {}] loss: {:.10f}".format(epoch, avg_loss)
            if epoch % update_interval == 0:
                logging.info(msg)
            else:
                logging.debug(msg)

        # write plot of loss over time
        # write_losses(self.losses, self.folder)
        write_figure(dataseries=self.losses, filename=f"{self.folder}/lossplot.png", title="Average Loss over Time", xlabel="epoch", ylabel="average loss")
