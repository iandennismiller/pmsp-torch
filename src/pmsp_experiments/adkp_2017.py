#!/usr/bin/env python3

# PMSP Torch
# Ian Dennis Miller, Brian Lam, Blair Armstrong

import logging

import torch
import torch.optim as optim
from torchsummary import summary

from pmsp.stimuli import build_stimuli_df, build_dataloader, append_dataloader
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

anchors_dataloader = build_dataloader(
    mapping_filename=f"{pmsp_path}/data/anchors.csv",
    frequency_filename=f"{pmsp_path}/data/word-frequencies.csv"
)

probes_dataloader = build_dataloader(
    mapping_filename=f"{pmsp_path}/data/probes.csv",
    frequency_filename=f"{pmsp_path}/data/word-frequencies.csv"
)

english_phonemes = Phonemes()

def train_base_vocabulary(do_training, trainer):
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
        trainer.network.save(filename="var/pmsp-base-vocabulary.zip")
    else:
        trainer.network.load(filename="var/pmsp-base-vocabulary.zip")


def train_anchors(do_training, trainer):
    with_anchors_dataloader = append_dataloader(anchors_dataloader, base_vocabulary_dataloader)

    # train for another 350 epochs
    if do_training == True:
        trainer.train(
            dataloader=with_anchors_dataloader,
            num_epochs=350,
            optimizers={0: optim.Adam(trainer.network.parameters(), lr=0.01)}
        )
        trainer.network.save(filename="var/adkp-with-anchors.zip")
    else:
        trainer.network.load(filename="var/adkp-with-anchors.zip")


def test_probes(network):
    step_idx, (frequency, graphemes, phonemes) = enumerate(probes_dataloader).__next__()

    outputs = network(graphemes)
    outputs_max_vowel = outputs[:, 23:37].argmax(dim=1).tolist()
    outputs_phonemes = [english_phonemes.one_hot_to_phoneme('vowel', x) for x in outputs_max_vowel]

    print(dict(zip(probes_dataloader.dl.dataset.df["orth"], outputs_phonemes)))


def main(train=False):
    torch.manual_seed(1)
    folder = make_folder()

    network = PMSPNetwork()
    trainer = PMSPTrainer(network=network)

    ###
    # Train base vocabulary

    train_base_vocabulary(do_training=train, trainer=trainer)

    write_figure(
        dataseries=trainer.losses,
        filename=f"{folder}/lossplot-base-vocabulary.png",
        title="Base vocabulary: average Loss over Time",
        xlabel="epoch",
        ylabel="average loss"
    )

    ###
    # Now include anchors in training data

    train_anchors(do_training=train, trainer=trainer)

    write_figure(
        dataseries=trainer.losses,
        filename=f"{folder}/lossplot-anchors.png",
        title="Add anchors: Average Loss over Time",
        xlabel="epoch",
        ylabel="average loss"
    )

    ###
    # Now test with probes

    test_probes(trainer.network)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO
    )
    main()
