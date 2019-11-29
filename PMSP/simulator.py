# PMSP Torch
# CAP Lab

from .model import PMSPNet, PMSPDoubleNet
from .util import make_folder, write_losses

import logging
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

class Simulator:
    def __init__(self, model, batch_size=None, num_workers=None, deterministic=True):
        if deterministic:
            torch.manual_seed(1)

        self.folder = make_folder()
        self.model = model

        if torch.cuda.is_available():
            logging.info("using CUDA")
            self.model.cuda()
        else:
            logging.info("using CPU")

    def go(self, learning_rate=0.001, update_interval=10, num_epochs=300):
        self.losses = []

        for epoch in range(num_epochs):
            avg_loss = self.model.train(learning_rate)
            self.losses.append(avg_loss)

            msg = "[EPOCH {}] loss: {:.10f}".format(epoch+1, avg_loss)
            if epoch % update_interval == 0:
                logging.info(msg)
            else:
                logging.debug(msg)

        # write plot of loss over time
        write_losses(self.losses, self.folder)
