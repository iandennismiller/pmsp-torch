#!/usr/bin/env python

# PMSP Torch
# CAP Lab

import sys
sys.path.insert(0, '.')
from PMSP.stimuli import PMSPStimuli
from PMSP.simulator import Simulator

import os
import click
from torchsummary import summary

@click.group()
def cli():
    pass

@click.command('test', short_help='Just test whether it runs.')
def just_test():
    stimuli = PMSPStimuli()
    result = stimuli.generate_stimuli(percentage=0.95)
    assert(result)

    sim = Simulator()
    summary(sim.model, input_size=(1, 1, sim.model.input_size))
    sim.train(num_epochs=3)

@click.command('generate', short_help='Generate data.')
@click.option('--write/--no-write', default=False, help='Write to file.')
def generate(write):
    stimuli = PMSPStimuli()
    result = stimuli.generate_stimuli(percentage=0.95)

    if write:
        path = os.getcwd()
        if not os.path.isdir(os.path.join(path, 'var')):
            os.mkdir(os.path.join(path, 'var'))
        if not os.path.isdir(os.path.join(path, 'var', 'stimuli')):
            os.mkdir(os.path.join(path, 'var', 'stimuli'))

        with open('var/stimuli/pmsp-regular-train.ex', 'w') as f:
            f.write(result['training'])
        with open('var/stimuli/pmsp-regular-test.ex', 'w') as f:
            f.write(result['testing'])
    else:
        print(result['training'])
        print(result['testing'])

@click.command('simulate', short_help='Run simulation training.')
@click.option('--rate', default=0.001, help='Learning rate.')
@click.option('--epochs', default=300, help='Number of epochs.')
def simulate(rate, epochs):
    sim = Simulator()
    sim.train(learning_rate=rate, num_epochs=epochs)
    # print(sim.model.layer2.bias.data)

cli.add_command(generate)
cli.add_command(just_test)
cli.add_command(simulate)

if __name__ == '__main__':
    from PMSP.__meta__ import __version__
    print("pmsp-torch", __version__)
    print("")
    cli()
