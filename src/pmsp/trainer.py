# PMSP Torch
# Ian Dennis Miller, Brian Lam, Blair Armstrong

import logging
import torch
import torch.nn as nn


class PMSPTrainer:
    def __init__(self, network, update_interval=10):
        self.network = network
        self.epoch = 0
        self.update_interval = update_interval
        self.accuracy = []
        self.losses = []

    def log_msg(self, msg):
        if self.epoch % self.update_interval == 0:
            logging.info(msg)
        else:
            logging.debug(msg)

    def train_one(self, dataloader):
        avg_loss = 0

        # DataLoader's batch size is equal to the entire trainind set.
        # Therefore, there is only one element to enumerate
        for step_idx, (frequency, graphemes, phonemes) in enumerate(dataloader):
            assert step_idx is not None

            # forward pass
            outputs = self.network(graphemes)

            # calculate loss
            loss = self.criterion(outputs, phonemes)
            loss = (loss * frequency.view(-1, 1)).mean()
            avg_loss += loss.item()

            # calculate accuracy
            outputs_max_vowel = outputs[:, 23:37].argmax(dim=1)
            targets_max_vowel = phonemes[:, 23:37].argmax(dim=1)
            compare = torch.eq(outputs_max_vowel, targets_max_vowel).tolist()
            
            # since there is just one batch per epoch, we can compute accuracy here
            self.accuracy.append(sum(compare) / len(outputs))
            # self.accuracy_per_epoch.append(dict(zip(dataloader.dl.dataset.df["orth"], compare)))

            # backprop
            loss.backward()

            # optimize
            self.optimizer.step()
            self.optimizer.zero_grad()
        
        # create record of loss per epoch
        avg_loss = avg_loss / len(dataloader)
        return(avg_loss)


    def train(self, dataloader, num_epochs, optimizers):
        self.criterion = nn.BCELoss(reduction='none')

        for epoch_iterator in range(num_epochs):
            # switch optimizer relative to epoch_iterator (not epoch)
            if epoch_iterator in optimizers:
                self.optimizer = optimizers[epoch_iterator]
                logging.info(f"switch to {self.optimizer}")

            # train one epoch and store the average loss
            avg_loss = self.train_one(dataloader)
            self.losses.append(avg_loss)
            latest_accuracy = self.accuracy[-1]

            self.log_msg(f"sim epoch: {self.epoch}; train epoch: {epoch_iterator}; loss: {avg_loss:.10f}; accuracy: {latest_accuracy:.10f}")
            self.epoch += 1

        return self.losses
