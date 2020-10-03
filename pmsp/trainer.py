# PMSP Torch
# Ian Dennis Miller, Brian Lam, Blair Armstrong

import logging
import torch
import torch.nn as nn
import torch.optim as optim

from .util import write_losses, make_folder


class PMSPTrainer:
    def __init__(self, network, dataloader, learning_rate=0.001):

        self.network = network
        self.dataloader = dataloader
        self.learning_rate = learning_rate

        if torch.cuda.is_available():
            logging.info("using CUDA")
            self.network.cuda()
        else:
            logging.info("using CPU")


    def train_one(self):
        avg_loss = 0

        for step_idx, (frequency, graphemes, phonemes) in enumerate(self.dataloader):

            freq = frequency.float().view(-1, 1)
            inputs = graphemes.float()
            targets = phonemes.float()
            
            # forward pass
            outputs = self.network(inputs)

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


    def train(self, update_interval=10, num_epochs=300):

        self.folder = make_folder()

        self.losses = []

        self.criterion = nn.BCELoss(reduction='none')
        self.optimizer = optim.Adam(self.network.parameters(), lr=self.learning_rate)

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
