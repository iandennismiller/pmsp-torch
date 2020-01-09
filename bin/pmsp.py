#!/usr/bin/env python

# PMSP Torch
# CAP Lab

import sys
sys.path.insert(0, '.')
from PMSP.stimuli import PMSPStimuli
from PMSP.model import PMSPNet
from PMSP.simulator import Simulator

import os
import click
import logging
from torchsummary import summary

@click.group()
def cli():
    pass

@click.command('test', short_help='Just test whether it runs.')
def just_test():
    stimuli = PMSPStimuli()
    result = stimuli.generate_stimuli(percentage=0.95)
    assert(result)

    sim = Simulator(model=PMSPNet())
    summary(sim.model, input_size=(1, 1, sim.model.input_size))
    sim.go(num_epochs=3)

@click.command('generate', short_help='Generate data.')
@click.option('--write/--no-write', default=False, help='Write to file.')
@click.option('--infile', default='pmsp-data.csv', help='File to read from.')
@click.option('--outfile', default=None, help='File to write to.')
def generate(write, infile, outfile):
    stimuli = PMSPStimuli(infile)
    result = stimuli.generate_stimuli()

    if write:
        path = os.getcwd()
        if not os.path.isdir(os.path.join(path, 'var')):
            os.mkdir(os.path.join(path, 'var'))
        if not os.path.isdir(os.path.join(path, 'var', 'stimuli')):
            os.mkdir(os.path.join(path, 'var', 'stimuli'))

        if not outfile:
            outfile = infile

        outfilename = os.path.join(path, 'var', 'stimuli', outfile)
        with open(outfilename, 'w') as f:
            f.write(result)
    else:
        print(result)

@click.command('simulate', short_help='Run simulation training.')
@click.option('--rate', default=0.001, help='Learning rate.')
@click.option('--epochs', default=300, help='Number of epochs.')
def simulate(rate, epochs):
    sim = Simulator(model=PMSPNet(learning_rate=rate))
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
