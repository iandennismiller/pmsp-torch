# PMSP Torch
# Ian Dennis Miller, Brian Lam, Blair Armstrong

import torch
import torch.optim as optim
from torchsummary import summary

from ..stimuli import build_stimuli_df, build_dataloader
from ..trainer import PMSPTrainer
from ..util import write_figure, make_folder
from . import StandardModel


class InspectVowelActivation(StandardModel):

    def go(self, retrain):
        torch.manual_seed(1)

        # set up the PMSP trainer to use this network and dataloader
        self.trainer = PMSPTrainer(network=self.network)

        optimizers = {
            0: optim.SGD(self.network.parameters(), lr=0.0001),
            10: optim.Adam(self.network.parameters(), lr=0.01)
        }

        losses = []

        # run for 350 epochs
        if retrain == True:
            losses = self.trainer.train(
                dataloader=self.pmsp_dataloader,
                num_epochs=350,
                optimizers=optimizers
            )
            self.network.save(filename="var/network-default.zip")
        else:
            self.network.load(filename="var/network-default.zip")

        # write plot of loss over time
        folder = make_folder()
        write_figure(dataseries=losses, filename=f"{folder}/lossplot.png", title="Average Loss over Time", xlabel="epoch", ylabel="average loss")

        self.adkp_probes, self.adkp_probes_dataset, self.adkp_probes_dataloader = build_dataloader(
            mapping_filename="pmsp/data/probes.csv",
            frequency_filename="pmsp/data/word-frequencies.csv"
        )

        step_idx, (frequency, graphemes, phonemes) = enumerate(self.adkp_probes_dataloader).__next__()

        outputs = self.network(graphemes)
        outputs_max_vowel = outputs[:, 23:37].argmax(dim=1).tolist()
        print(outputs)
        print(outputs_max_vowel)
