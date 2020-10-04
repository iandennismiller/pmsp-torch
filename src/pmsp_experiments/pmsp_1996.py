#!/usr/bin/env python3

# PMSP Torch
# Ian Dennis Miller, Brian Lam, Blair Armstrong

import os
import logging

import torch
import torch.optim as optim

from pmsp.network import PMSPNetwork
from pmsp.trainer import PMSPTrainer
from pmsp.stimuli import build_stimuli_df, build_dataloader
from pmsp.util import write_figure, make_folder, get_pmsp_path


def main(train=False):
    pmsp_path = get_pmsp_path()

    pmsp_stimuli, pmsp_dataset, pmsp_dataloader = build_dataloader(
        mapping_filename=f"{pmsp_path}/data/plaut_dataset_collapsed.csv",
        frequency_filename=f"{pmsp_path}/data/word-frequencies.csv"
    )

    torch.manual_seed(1)

    network = PMSPNetwork()
    trainer = PMSPTrainer(network=network)

    optimizers = {
        0: optim.SGD(network.parameters(), lr=0.0001),
        10: optim.Adam(network.parameters(), lr=0.01)
    }

    losses = trainer.train(
        dataloader=pmsp_dataloader,
        num_epochs=350,
        optimizers=optimizers
    )

    folder = make_folder()
    write_figure(
        dataseries=losses,
        filename=f"{folder}/lossplot.png",
        title="Average Loss over Time",
        xlabel="epoch",
        ylabel="average loss"
    )


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO
    )
    main()
