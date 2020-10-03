# PMSP Torch
# Ian Dennis Miller, Brian Lam, Blair Armstrong

import pandas as pd
from math import log

from .english.phonemes import Phonemes
from .english.graphemes import Graphemes
from .english.frequencies import Frequencies

from pmsp.dataset import PMSPDataset
from pmsp.util.device_dataloader import DeviceDataLoader
from torch.utils.data import DataLoader


# do not warn about assignment to a copy of a pandas object
pd.options.mode.chained_assignment = None


def build_stimuli_df(mapping_filename, frequency_filename):
    """
    This takes a CSV file with a mapping from orthography to phonology in it.
    It produces a data frame containing one-hot encodings of the mapping.
    Optionally, this can also look up word occurrence frequencies.
    """

    # load graphemes, phonemes, and word frequencies
    graphemes = Graphemes()
    phonemes = Phonemes()
    frequencies = Frequencies(frequency_filename)

    df = pd.read_csv(mapping_filename, sep=",")

    # If there is a column called "freq" in the dataset, then use that
    try:
        freqs = df["freq"]
        lookup_frequencies = False
    # otherwise look up the frequencies in a reference CSV
    except:
        lookup_frequencies = True

    # drop all other columns apart from orth, phon, and type
    df = df[["orth", "phon", "type"]]

    # create new grapheme one-hot encoded column by converting orthographies
    df["graphemes"] = df["orth"].apply(
        lambda x: graphemes.get_graphemes(x)
    )

    # create new phoneme one-hot encoded column by converting phonologies
    df["phonemes"] = df["phon"].apply(
        lambda x: phonemes.get_phonemes(x)
    )

    # assign frequencies if this was previously specified
    if lookup_frequencies:
        df["frequency"] = df["orth"].apply(
            lambda x: frequencies.get_frequency(x)
        )
    else:
        df["frequency"] = freqs

    # standardize frequency
    # freq_sum = df["frequency"].sum()
    # df["frequency"] = df["frequency"].apply(
    #     lambda x: (x / freq_sum)
    # )

    # log transform frequency
    df["frequency"] = df["frequency"].apply(
        lambda x: log(x+2)
    )

    return df

def build_dataloader(mapping_filename, frequency_filename):
    # stimuli are drawn from these CSV files
    stimuli = build_stimuli_df(
        mapping_filename=mapping_filename,
        frequency_filename=frequency_filename
    )

    # build dataset from stimuli
    dataset = PMSPDataset(stimuli)

    # build dataloader from dataset
    dataloader = DeviceDataLoader(DataLoader(
        dataset,
        batch_size=len(dataset),
        num_workers=0
    ))

    return stimuli, dataset, dataloader
