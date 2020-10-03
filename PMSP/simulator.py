# PMSP Torch
# Ian Dennis Miller, Brian Lam, Blair Armstrong

import torch
import logging

from .network import PMSPNetwork
from .trainer import PMSPTrainer
from .util import make_folder


class Simulator:
    def __init__(self, network):
        if torch.cuda.is_available():
            logging.info("using CUDA")
            network.cuda()
        else:
            logging.info("using CPU")

        self.folder = make_folder()
        self.network = network

    def go(self, batch_size=None, num_workers=None, deterministic=True):
        if deterministic:
            torch.manual_seed(1)

        trainer = PMSPTrainer()
        trainer.train_all()
