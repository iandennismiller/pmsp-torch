#!/usr/bin/env python

# PMSP Torch
# Ian Dennis Miller, Brian Lam, Blair Armstrong

import os
import click
import logging
from torchsummary import summary

import sys
sys.path.insert(0, '.')

from PMSP.stimuli import PMSPStimuli
from PMSP.network import PMSPNetwork
from PMSP.simulator import Simulator


@click.group()
def cli():
    pass

@click.command('test', short_help='Just test whether it runs.')
def just_test():
    stimuli = PMSPStimuli()
    result = stimuli.generate_stimuli_log_transform(percentage=0.95)
    assert(result)

    sim = Simulator(model=PMSPNetwork())
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
    sim = Simulator(model=PMSPNetwork(learning_rate=rate))
    sim.go(num_epochs=epochs)

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
