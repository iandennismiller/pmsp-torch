#!/usr/bin/env python

# PMSP Torch
# Ian Dennis Miller, Brian Lam, Blair Armstrong

import os
import click
import logging
from torchsummary import summary
from torch.utils.data import DataLoader

import sys
sys.path.insert(0, '.')

from pmsp.stimuli import PMSPStimuli
from pmsp.dataset import PMSPDataset
from pmsp.device_dataloader import DeviceDataLoader
from pmsp.network import PMSPNetwork
from pmsp.trainer import PMSPTrainer
from pmsp.simulator import Simulator

@click.group()
def cli():
    pass

@click.command('test', short_help='Just test whether it runs.')
def just_test():
    stimuli = PMSPStimuli(
        mapping_filename="PMSP/data/plaut_dataset_collapsed.csv",
        frequency_filename="PMSP/data/word-frequencies.csv",
        )
    dataset = PMSPDataset(stimuli.df)
    network = PMSPNetwork(dataset=dataset)

    # stimuli = PMSPStimuli()
    # result = stimuli.generate_stimuli_log_transform(percentage=0.95)
    # assert(result)

    sim = Simulator(model=network)
    summary(sim.model, input_size=(1, 1, sim.model.input_size))
    sim.go(num_epochs=3)

@click.command('generate', short_help='Generate data.')
@click.option('--infile', required=True, help='File to read from.')
@click.option('--outfile', required=True, help='File to write to.')
def generate(infile, outfile):
    stimuli = PMSPStimuli(infile)
    result = stimuli.generate_stimuli()
    with open(outfile, 'w') as f:
        f.write(result)

@click.command('simulate', short_help='Run simulation training.')
@click.option('--rate', default=0.001, help='Learning rate.')
@click.option('--epochs', default=300, help='Number of epochs.')
def simulate(rate, epochs):
    mapping_filename = "PMSP/data/plaut_dataset_collapsed.csv"
    frequency_filename = "PMSP/data/word-frequencies.csv"

    stimuli = PMSPStimuli(
        mapping_filename=mapping_filename,
        frequency_filename=frequency_filename
    )
    dataset = PMSPDataset(stimuli.df)
    dataloader = DeviceDataLoader(DataLoader(
        dataset,
        batch_size=len(dataset),
        num_workers=0
    ))

    network = PMSPNetwork()
    trainer = PMSPTrainer(network=network, dataloader=dataloader)

    sim = Simulator()
    sim.go(trainer, num_epochs=epochs)

cli.add_command(generate)
cli.add_command(just_test)
cli.add_command(simulate)

if __name__ == '__main__':
    if not os.path.isdir('var'):
        os.mkdir('var')

    logging.basicConfig(
        filename='var/pmsp.log',
        level=logging.INFO
    )

    from PMSP.__meta__ import __version__
    print("pmsp-torch", __version__)
    cli()
