# PMSP Torch
# Ian Dennis Miller, Brian Lam, Blair Armstrong

import torch
import logging


class Simulator:
    def __init__(self):
        pass

    def go(self, trainer, num_epochs, deterministic=True):
        if deterministic:
            torch.manual_seed(1)

        trainer.train(num_epochs=num_epochs)
