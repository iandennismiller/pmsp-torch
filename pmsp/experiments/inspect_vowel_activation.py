# PMSP Torch
# Ian Dennis Miller, Brian Lam, Blair Armstrong

import os
import click
import logging
import torch
import torch.optim as optim
from torchsummary import summary
from torch.utils.data import DataLoader

import sys
sys.path.insert(0, '.')

from pmsp.stimuli import build_stimuli_df
from pmsp.dataset import PMSPDataset
from pmsp.util.device_dataloader import DeviceDataLoader
from pmsp.network import PMSPNetwork
from pmsp.trainer import PMSPTrainer

from . import Experiment


class InspectVowelActivation(Experiment):

    def __init__(self):
        torch.manual_seed(1)

        # stimuli are drawn from these CSV files
        self.stimuli = build_stimuli_df(
            mapping_filename="pmsp/data/plaut_dataset_collapsed.csv",
            frequency_filename="pmsp/data/word-frequencies.csv"
        )

        # build dataset from stimuli
        self.dataset = PMSPDataset(self.stimuli)

        # build dataloader from dataset
        self.dataloader = DeviceDataLoader(DataLoader(
            self.dataset,
            batch_size=len(self.dataset),
            num_workers=0
        ))

        # set up the network
        self.network = PMSPNetwork()
        summary(self.network, input_size=(1, 1, self.network.input_size))

        # set up the PMSP trainer to use this network and dataloader
        self.trainer = PMSPTrainer(network=self.network, dataloader=self.dataloader)


    def go(self, retrain):
        optimizers = {
            0: optim.SGD(self.network.parameters(), lr=0.0001),
            10: optim.Adam(self.network.parameters(), lr=0.01)
        }

        # run for 350 epochs
        if retrain == True:
            self.trainer.train(num_epochs=350, optimizers=optimizers)
            self.network.save(filename="var/network-default.zip")
        else:
            self.network.load(filename="var/network-default.zip")

        step_idx, (frequency, graphemes, phonemes) = enumerate(self.dataloader).__next__()
        print(graphemes)

        outputs = self.network(graphemes)
        outputs_max_vowel = outputs[:, 23:37].argmax(dim=1).tolist()
        print(outputs)
        print(outputs_max_vowel)
