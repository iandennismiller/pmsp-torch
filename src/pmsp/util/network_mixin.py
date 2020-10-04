# PMSP Torch
# Ian Dennis Miller, Brian Lam, Blair Armstrong

import logging
import torch

class NetworkMixin:
    def init_cuda(self):
        if torch.cuda.is_available():
            logging.info("using CUDA")
            self.cuda()
        else:
            logging.info("using CPU")

    def save(self, filename):
        torch.save(self.state_dict(), filename)

    def load(self, filename):
        self.load_state_dict(torch.load(filename))
        self.eval()
