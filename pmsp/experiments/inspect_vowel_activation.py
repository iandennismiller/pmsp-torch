# PMSP Torch
# Ian Dennis Miller, Brian Lam, Blair Armstrong

import os
import click
import logging
import torch
from torchsummary import summary
from torch.utils.data import DataLoader

import sys
sys.path.insert(0, '.')

from pmsp.stimuli import PMSPStimuli
from pmsp.dataset import PMSPDataset
from pmsp.device_dataloader import DeviceDataLoader
from pmsp.network import PMSPNetwork
from pmsp.trainer import PMSPTrainer


def go(retrain):
    torch.manual_seed(1)

    # stimuli are drawn from these CSV files
    stimuli = PMSPStimuli(
        mapping_filename="pmsp/data/plaut_dataset_collapsed.csv",
        frequency_filename="pmsp/data/word-frequencies.csv"
    )

    # build dataset from stimuli
    dataset = PMSPDataset(stimuli.df)

    # build dataloader from dataset
    dataloader = DeviceDataLoader(DataLoader(
        dataset,
        batch_size=len(dataset),
        num_workers=0
    ))

    # set up the network
    network = PMSPNetwork()
    summary(network, input_size=(1, 1, network.input_size))

    # set up the PMSP trainer to use this network and dataloader
    trainer = PMSPTrainer(network=network, dataloader=dataloader)

    # run for 350 epochs
    if retrain == True:
        trainer.train(num_epochs=350)
        network.save(filename="var/network-default.zip")
    else:
        network.load(filename="var/network-default.zip")

    step_idx, (frequency, graphemes, phonemes) = enumerate(dataloader).__next__()
    print(graphemes)

    outputs = network(graphemes)
    outputs_max_vowel = outputs[:, 23:37].argmax(dim=1).tolist()
    print(outputs)
    print(outputs_max_vowel)
