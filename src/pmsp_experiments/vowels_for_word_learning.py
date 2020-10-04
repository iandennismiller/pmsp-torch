#!/usr/bin/env python3

# PMSP Torch
# Ian Dennis Miller, Brian Lam, Blair Armstrong

import logging
import torch
import torch.optim as optim
from torchsummary import summary

from pmsp.stimuli import build_stimuli_df, build_dataloader
from pmsp.trainer import PMSPTrainer
from pmsp.network import PMSPNetwork
from pmsp.util import write_figure, make_folder


def main(retrain=False):
    torch.manual_seed(1)

    pmsp_stimuli, pmsp_dataset, pmsp_dataloader = build_dataloader(
        mapping_filename="pmsp/data/plaut_dataset_collapsed.csv",
        frequency_filename="pmsp/data/word-frequencies.csv"
    )

    network = PMSPNetwork()
    trainer = PMSPTrainer(network=network)

    optimizers = {
        0: optim.SGD(network.parameters(), lr=0.0001),
        10: optim.Adam(network.parameters(), lr=0.01)
    }

    losses = []

    # run for 350 epochs
    if retrain == True:
        losses = trainer.train(
            dataloader=pmsp_dataloader,
            num_epochs=350,
            optimizers=optimizers
        )
        network.save(filename="var/network-default.zip")
    else:
        network.load(filename="var/network-default.zip")

    # write plot of loss over time
    folder = make_folder()
    write_figure(
        dataseries=losses,
        filename=f"{folder}/lossplot.png",
        title="Average Loss over Time",
        xlabel="epoch",
        ylabel="average loss"
    )

    adkp_probes, adkp_probes_dataset, adkp_probes_dataloader = build_dataloader(
        mapping_filename="pmsp/data/probes.csv",
        frequency_filename="pmsp/data/word-frequencies.csv"
    )

    step_idx, (frequency, graphemes, phonemes) = enumerate(adkp_probes_dataloader).__next__()

    outputs = network(graphemes)
    outputs_max_vowel = outputs[:, 23:37].argmax(dim=1).tolist()
    print(outputs)
    print(outputs_max_vowel)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO
    )
    main()
