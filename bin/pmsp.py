#!/usr/bin/env python

# PMSP Torch
# Ian Dennis Miller, Brian Lam, Blair Armstrong

import os
import sys
import click
import logging

sys.path.insert(0, '.')

@click.group()
def cli():
    pass


@click.command('generate', short_help='Generate data.')
@click.option('--wordfile', required=True, help='Word file to read from.')
@click.option('--freqfile', required=True, help='Frequency file to read from.')
@click.option('--outfile', required=True, help='File to write to.')
def generate(wordfile, freqfile, outfile):
    from pmsp.stimuli import build_stimuli_df
    from pmsp.util.lens_stimuli import generate_stimuli
    stimuli_df = build_stimuli_df(wordfile, freqfile)
    result = generate_stimuli(stimuli_df)
    with open(outfile, 'w') as f:
        f.write(result)


@click.command('inspect-vowel-activation')
@click.option('--retrain/--no-retrain', default=False)
def inspect_vowel_activation(retrain):
    from pmsp_experiments.inspect_vowel_activation import main
    main(retrain)


@click.command('adkp-2017')
@click.option('--retrain/--no-retrain', default=False)
def replicate_adkp_2017(retrain):
    from pmsp_experiments.replicate_adkp_2017 import main
    main(retrain)


@click.command('pmsp-1996')
@click.option('--retrain/--no-retrain', default=False)
def replicate_pmsp_1996(retrain):
    from pmsp_experiments.replicate_pmsp_1996 import main
    main(retrain)


cli.add_command(replicate_adkp_2017)
cli.add_command(replicate_pmsp_1996)
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
