# PMSP Torch
# Ian Dennis Miller, Brian Lam, Blair Armstrong

import re
import os
import copy
import torch
import random
import inspect

from math import log
import pandas as pd

from torch.utils.data import Dataset

# do not warn about assignment to a copy of a pandas object
pd.options.mode.chained_assignment = None

class PMSPDataset(Dataset):
    def __init__(self, df):
        self.frequency_tensor = torch.tensor(df["frequency"])
        self.grapheme_tensor = torch.tensor(df["graphemes"])
        self.phoneme_tensor = torch.tensor(df["phonemes"])
        self.num_samples = len(df)

    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, index):
        return (
            self.frequency_tensor[index],
            self.grapheme_tensor[index],
            self.phoneme_tensor[index]
        )
