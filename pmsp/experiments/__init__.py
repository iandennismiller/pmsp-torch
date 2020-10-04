# PMSP Torch
# Ian Dennis Miller, Brian Lam, Blair Armstrong

from pmsp.stimuli import build_dataloader
from pmsp.network import PMSPNetwork


class StandardModel:

    def __init__(self):
        # set up the network
        self.network = PMSPNetwork()

        self.pmsp_stimuli, self.pmsp_dataset, self.pmsp_dataloader = build_dataloader(
            mapping_filename="pmsp/data/plaut_dataset_collapsed.csv",
            frequency_filename="pmsp/data/word-frequencies.csv"
        )


def calculate_accuracy(data):
    # https://github.com/Brian0615/Plaut_Model_v2/blob/0b3009b20ed366f6bdc9bd262c7bdd7f3869f33f/simulator/simulator.py#L434

    # accuracy calculations - accuracy is based on highest activity vowel phoneme

    inputs = data['graphemes']
    targets = data['phonemes'].clone()

    outputs_max_vowel = outputs[:, 23:37].argmax(dim=1)
    targets_max_vowel = targets[:, 23:37].argmax(dim=1)
    compare = torch.eq(outputs_max_vowel, targets_max_vowel).tolist()

    correct, total = [], []
    # find correct and total for each category
    for cat in categories:
        if cat == 'All':  # find accuracy across all categories
            correct.append(sum(compare) / len(data['type']))
            total.append(len(data['type']))
        else:
            temp_total, temp_correct = 0, 0
            for t, c in zip(data['type'], compare):
                if t == cat:  # if same category
                    temp_total += 1
                    if c:  # if correct
                        temp_correct += 1
            total.append(temp_total)
            correct.append(temp_correct)
