# PMSP Torch
# Ian Dennis Miller, Brian Lam, Blair Armstrong

import torch
from torch.utils.data import Dataset


class PMSPDataset(Dataset):
    def __init__(self, df):
        # store stimuli_df for later
        self.df = df

        self.frequency_tensor = torch.Tensor(df["frequency"]).float()
        self.grapheme_tensor = torch.Tensor(df["graphemes"]).float()
        self.phoneme_tensor = torch.Tensor(df["phonemes"]).float()
        self.num_samples = len(df)

    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, index):
        return (
            self.frequency_tensor[index],
            self.grapheme_tensor[index],
            self.phoneme_tensor[index],
        )
