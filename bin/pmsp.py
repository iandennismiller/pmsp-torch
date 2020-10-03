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


@click.group()
def cli():
    pass


@click.command('generate', short_help='Generate data.')
@click.option('--wordfile', required=True, help='Word file to read from.')
@click.option('--freqfile', required=True, help='Frequency file to read from.')
@click.option('--outfile', required=True, help='File to write to.')
def generate(wordfile, freqfile, outfile):
    from pmsp.stimuli import PMSPStimuli
    stimuli = PMSPStimuli(wordfile, freqfile)
    result = stimuli.generate_stimuli()
    with open(outfile, 'w') as f:
        f.write(result)


@click.command('inspect-vowel-activation')
@click.option('--retrain/--no-retrain', default=False, help='Whether to retrain the network.')
def inspect_vowel_activation(retrain):
    from pmsp.experiments.inspect_vowel_activation import InspectVowelActivation

    experiment = InspectVowelActivation()
    experiment.go(retrain)


cli.add_command(generate)
cli.add_command(inspect_vowel_activation)

if __name__ == '__main__':
    if not os.path.isdir('var'):
        os.mkdir('var')

    logging.basicConfig(
        filename='var/pmsp.log',
        level=logging.INFO
    )

    from pmsp.__meta__ import __version__
    print("pmsp-torch", __version__)
    cli()
