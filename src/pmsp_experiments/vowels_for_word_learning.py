#!/usr/bin/env python3

# PMSP Torch
# Ian Dennis Miller, Brian Lam, Blair Armstrong

import logging
import torch
import torch.optim as optim
from torchsummary import summary

import pandas as pd

from pmsp.stimuli import build_stimuli_df, build_dataloader
from pmsp.trainer import PMSPTrainer
from pmsp.network import PMSPNetwork
from pmsp.util import write_figure, make_folder, get_pmsp_path
from pmsp.english.phonemes import Phonemes


###
# Prepare data loaders with training examples
# These are global with respect to training procedures below

pmsp_path = get_pmsp_path()

base_vocabulary_dataloader = build_dataloader(
    mapping_filename=f"{pmsp_path}/data/plaut_dataset_collapsed.csv",
    frequency_filename=f"{pmsp_path}/data/word-frequencies.csv"
)

behavioural_stimuli_dataloader = build_dataloader(
    mapping_filename=f"{pmsp_path}/data/behavioural-stimuli.csv",
    frequency_filename=f"{pmsp_path}/data/word-frequencies.csv"
)

english_phonemes = Phonemes()

def train_base_vocabulary(do_training, trainer, seed):
    optimizers = {
        0: optim.SGD(trainer.network.parameters(), lr=0.0001),
        10: optim.Adam(trainer.network.parameters(), lr=0.01)
    }

    # run for 350 epochs
    if do_training == True:
        trainer.train(
            dataloader=base_vocabulary_dataloader,
            num_epochs=350,
            optimizers=optimizers
        )
        trainer.network.save(filename=f"var/pmsp-base-vocabulary-{seed}.zip")
    else:
        trainer.network.load(filename=f"var/pmsp-base-vocabulary-{seed}.zip")


def test_behavioural_stimuli(network):
    step_idx, (frequency, graphemes, phonemes) = enumerate(behavioural_stimuli_dataloader).__next__()

    outputs = network(graphemes)
    outputs_max_vowel = outputs[:, 23:37].argmax(dim=1).tolist()
    outputs_phonemes = [english_phonemes.one_hot_to_phoneme('vowel', x) for x in outputs_max_vowel]

    # outputs_phonemes = [x for x in outputs_max_vowel]

    # this computes the full output phonemes
    # outputs_phonemes = [english_phonemes.expand_one_hot(x) for x in outputs.tolist()]

    result = dict(zip(behavioural_stimuli_dataloader.dl.dataset.df["orth"], outputs_phonemes))

    return result


def main(train=False):
    vowels_df = pd.DataFrame(behavioural_stimuli_dataloader.dl.dataset.df["orth"], columns=["orth"])

    for seed in range(0, 30):
        torch.manual_seed(seed)

        network = PMSPNetwork()
        trainer = PMSPTrainer(network=network)

        ###
        # Train base vocabulary
        train_base_vocabulary(do_training=train, trainer=trainer, seed=seed)

        ###
        # Test with behavioural stimuli
        vowels = test_behavioural_stimuli(trainer.network)
        vowels_df[f"seed {seed}"] = vowels.values()

        print(vowels_df)

    vowels_df.to_csv('var/vowel-activations.csv')


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO
    )
    main()
