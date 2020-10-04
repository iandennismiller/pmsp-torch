# PMSP Torch
# Ian Dennis Miller, Brian Lam, Blair Armstrong

import sys
import logging

import torch
import torch.optim as optim

from ..stimuli import build_stimuli_df, build_dataloader
from ..network import PMSPNetwork
from ..trainer import PMSPTrainer
from ..util import write_figure, make_folder

logging.basicConfig(
  # filename='/content/drive/My Drive/Colab Notebooks/pmsp.log',
  level=logging.INFO
)

pmsp_stimuli, pmsp_dataset, pmsp_dataloader = build_dataloader(
  mapping_filename="pmsp/data/plaut_dataset_collapsed.csv",
  frequency_filename="pmsp/data/word-frequencies.csv"
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
write_figure(dataseries=losses, filename=f"{folder}/lossplot.png", title="Average Loss over Time", xlabel="epoch", ylabel="average loss")
