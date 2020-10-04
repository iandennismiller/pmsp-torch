# PMSP Torch
# Ian Dennis Miller, Brian Lam, Blair Armstrong

import os
import inspect
import datetime
import matplotlib.pyplot as plt


def get_pmsp_path():
    from ..network import PMSPNetwork
    src_file = inspect.getsourcefile(PMSPNetwork)
    pmsp_path = os.path.dirname(os.path.realpath(src_file))
    return pmsp_path


class StandardModel:

    def __init__(self):
        from ..stimuli import build_dataloader
        from ..network import PMSPNetwork

        self.network = PMSPNetwork()
        self.pmsp_path = get_pmsp_path()
        self.pmsp_stimuli, self.pmsp_dataset, self.pmsp_dataloader = build_dataloader(
            mapping_filename=f"{self.pmsp_path}/data/plaut_dataset_collapsed.csv",
            frequency_filename=f"{self.pmsp_path}/data/word-frequencies.csv"
        )


def plot_figure(dataseries, title, xlabel, ylabel):
    plt.figure()
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.plot(dataseries, label=title)
    return plt


def write_figure(dataseries, title, xlabel, ylabel, filename):
    plot_figure(dataseries, title, xlabel, ylabel)
    plt.savefig(filename, dpi=150)
    plt.close()


def make_folder():
    # create a new folder for each run
    now = datetime.datetime.now()
    date = now.strftime("%b").lower()+now.strftime("%d")

    path = os.getcwd()
    if not os.path.isdir(os.path.join(path, 'var')):
        os.mkdir(os.path.join(path, 'var'))
    if not os.path.isdir(os.path.join(path, 'var', 'results')):
        os.mkdir(os.path.join(path, 'var', 'results'))

    i = 1
    while True:
        try:
            rootdir = f'{path}/var/results/{date}_test{i:02d}'
            os.mkdir(rootdir)
            break
        except:
            i += 1

    return(rootdir)

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
