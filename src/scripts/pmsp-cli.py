#!/usr/bin/env python

# PMSP Torch
# Ian Dennis Miller, Brian Lam, Blair Armstrong

import os
import sys
import click
import logging

sys.path.insert(0, './src')

from pmsp.stimuli import build_stimuli_df
from pmsp.util.lens_stimuli import generate_stimuli, generate_stimuli_the_normalized

@click.group()
def cli():
    pass


@click.command('generate', short_help='Generate data.')
@click.option('--wordfile', required=True, help='Word file to read from.')
@click.option('--freqfile', required=True, help='Frequency file to read from.')
@click.option('--thenormalized/--no-thenormalized', default=False, help='Normalize frequencies to THE.')
@click.option('--outfile', required=True, help='File to write to.')
def lens_stimuli(wordfile, freqfile, thenormalized, outfile):
    stimuli_df = build_stimuli_df(wordfile, freqfile)

    # normalize frequencies to THE? (i.e. freq / 69971)
    if thenormalized:
        result = generate_stimuli_the_normalized(stimuli_df)
    else:
        result = generate_stimuli(stimuli_df)

    with open(outfile, 'w') as f:
        f.write(result)


@click.command('vowels-for-word-learning', short_help='How does the model regularize vowels?')
@click.option('--train/--no-train', default=False)
def vowels_for_word_learning(train):
    from pmsp_experiments.vowels_for_word_learning import main
    main(train)


@click.command('adkp-2017', short_help='Armstrong, Dumay, Kim, Pitt. (2017)')
@click.option('--train/--no-train', default=False)
def adkp_2017(train):
    from pmsp_experiments.adkp_2017 import main
    main(train)


@click.command('pmsp-1996', short_help='Plaut, McClelland, Seidenberg, Patterson. (1996)')
@click.option('--train/--no-train', default=False)
def pmsp_1996(train):
    from pmsp_experiments.pmsp_1996 import main
    main(train)


@click.command('mdlpa-2020', short_help='Miller, Dumay, Lam, Pitt, Armstrong. (2020)')
@click.option('--train/--no-train', default=False)
def mdlpa_2020(train):
    from pmsp_experiments.mdlpa_2020 import main
    main(train)


cli.add_command(lens_stimuli)
cli.add_command(pmsp_1996)
cli.add_command(adkp_2017)
cli.add_command(mdlpa_2020)
cli.add_command(vowels_for_word_learning)


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
