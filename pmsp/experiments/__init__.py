# PMSP Torch
# Ian Dennis Miller, Brian Lam, Blair Armstrong

import torch

import sys
sys.path.insert(0, '.')

from pmsp.stimuli import build_dataloader
from pmsp.network import PMSPNetwork


class StandardModel:

    def __init__(self):
        # set up the network
        self.network = PMSPNetwork()

        self.pmsp_stimuli, self.pmsp_dataset, self.pmsp_dataloader = build_dataloader(
            mapping_filename="pmsp/data/plaut_dataset_collapsed.csv",
            frequency_filename="pmsp/data/word-frequencies.csv"
        )

