# PMSP Torch
# CAP Lab

import os
import inspect
import pandas as pd
import torch
import random
import re
import copy
from torch.utils.data import Dataset

class PMSPDataset(Dataset):
    def __init__(self, df):
        self.df = df

    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, index):
        buf = {
            "type": self.df.loc[index, "type"],
            "orth": self.df.loc[index, "orth"],
            "phon": self.df.loc[index, "phon"],
            "frequency": torch.tensor(self.df.loc[index, "frequency"]),
            "graphemes": torch.tensor(self.df.loc[index, 'graphemes']),
            "phonemes": torch.tensor(self.df.loc[index, 'phonemes'])
        }

        return(buf)

class PMSPStimuli:
    def __init__(self):
        pathname = os.path.dirname(inspect.getfile(self.__class__))
        filename = os.path.join(pathname, 'data', 'pmsp-data.tsv')

        # load orthography and phonology
        self.mapper = PMSPOrthoPhonoMapping()
        self.df = pd.read_csv(filename, sep="\t")
        self.df = self.df[["orth", "phon", "type"]]
        self.df["graphemes"] = self.df["orth"].apply(
            lambda x: self.mapper.get_graphemes(x)
        )
        self.df["phonemes"] = self.df["phon"].apply(
            lambda x: self.mapper.get_phonemes(x)
        )

        # load word frequencies
        freq_file = os.path.join(pathname, 'data', 'freq.csv')
        df_freq = pd.read_csv(freq_file, header=0)
        self.frequencies = {}
        for index, row in df_freq.iterrows():
            self.frequencies[row['WORD']] = row['KFFRQ']

        # standardize frequency
        self.df["frequency"] = self.df["orth"].apply(
            lambda x: self.get_frequency(x)
        )
        freq_sum = self.df["frequency"].sum()
        self.df["frequency"] = self.df["frequency"].apply(
            lambda x: (x / freq_sum)
        )

        self.dataset = PMSPDataset(self.df)

    def generate_stimuli(self, percentage=0.5):
        buffer = {
            'training': "",
            'testing': "",
        }

        full_set = set(range(0, len(self.dataset)))
        training_set = set(random.sample(full_set, int(len(self.dataset) * percentage)))
        testing_set = full_set.difference(training_set)

        sets = [
            ('training', training_set),
            ('testing', testing_set)
        ]

        for current_buffer, current_set in sets:
            count = 0
            for idx in current_set:
                item = copy.copy(self.dataset[idx])
                item['phon'] = re.sub(r'\W', '', item['phon'])
                if item['type'] == '#': item['type'] = '0'
                buffer[current_buffer] += "name: {{{count}_{orth}_{phon}_{type}}}\n".format(count=count, **item)

                buffer[current_buffer] += "freq: {0:.9f}\n".format(item['frequency'])

                in_vec = ' '.join(str(x) for x in item['graphemes'].tolist())
                buffer[current_buffer] += "I: {}\n".format(in_vec)

                target_vec = ' '.join(str(x) for x in item['phonemes'].tolist())
                buffer[current_buffer] += "T: {};\n\n".format(target_vec)
                count += 1

        return(buffer)

    def get_frequency(self, word):
        word = str(word).upper()
        if word in self.frequencies:
            return(self.frequencies[word])
        else:
            return(1)

class PMSPOrthoPhonoMapping:
    def __init__(self):
        self.graphemes = {
            'onset': ['Y', 'S', 'P', 'T', 'K', 'Q', 'C', 'B', 'D', 'G', 'F', 'V', 'J', 'Z', 'L', 'M', 'N', 'R', 'W', 'H', 'CH', 'GH', 'GN', 'PH', 'PS', 'RH', 'SH', 'TH', 'TS', 'WH'],
            'vowel': ['E', 'I', 'O', 'U', 'A', 'Y', 'AI', 'AU', 'AW', 'AY', 'EA', 'EE', 'EI', 'EU', 'EW', 'EY', 'IE', 'OA', 'OE', 'OI', 'OO', 'OU', 'OW', 'OY', 'UE', 'UI', 'UY'],
            'coda': ['H', 'R', 'L', 'M', 'N', 'B', 'D', 'G', 'C', 'X', 'F', 'V', 'J', 'S', 'Z', 'P', 'T', 'K', 'Q', 'BB', 'CH', 'CK', 'DD', 'DG', 'FF', 'GG', 'GH', 'GN', 'KS', 'LL', 'NG', 'NN', 'PH', 'PP', 'PS', 'RR', 'SH', 'SL', 'SS', 'TCH', 'TH', 'TS', 'TT', 'ZZ', 'U', 'E', 'ES', 'ED'],
        }

        self.phonemes = {
            'onset': ['s', 'S', 'C', 'z', 'Z', 'j', 'f', 'v', 'T', 'D', 'p', 'b', 't', 'd', 'k', 'g', 'm', 'n', 'h', 'l', 'r', 'w', 'y'],
            'vowel': ['a', 'e', 'i', 'o', 'u', '@', '^', 'A', 'E', 'I', 'O', 'U', 'W', 'Y'],
            'coda': ['r', 'l', 'm', 'n', 'N', 'b', 'g', 'd', 'ps', 'ks', 'ts', 's', 'z', 'f', 'v', 'p', 'k', 't', 'S', 'Z', 'T', 'D', 'C', 'j'],
        }

    def get_phonemes(self, phon):
        # set initial phoneme vectors to zero
        onset = [0 for i in range(len(self.phonemes['onset']))]
        vowel = [0 for i in range(len(self.phonemes['vowel']))]
        codas = [0 for i in range(len(self.phonemes['coda']))]

        # drop first and last characters
        phon = phon[1:-1]

        # scan until first vowel, if any
        for i in range(len(phon)):
            if phon[i] in self.phonemes['vowel']:
                break
            # any phoneme onsets are coded in the onset vector
            if phon[i] in self.phonemes['onset']:
                onset[self.phonemes['onset'].index(phon[i])] = 1

        # scan for first coda, starting from the location of the first vowel
        for j in range(i, len(phon)):
            if phon[j] in self.phonemes['coda']:
                break
            # any phoneme vowels are coded in the vowel vector
            if phon[j] in self.phonemes['vowel']:
                vowel[self.phonemes['vowel'].index(phon[j])] = 1

        # starting from location of coda, set other codas in coda vector
        for k in range(j, len(phon)):
            # any phoneme codas are coded in the coda vector
            if phon[k] in self.phonemes['coda']:
                codas[self.phonemes['coda'].index(phon[k])] = 1
            # scan for 2-character codas, too
            if phon[k:k+2] in self.phonemes['coda']:
                codas[self.phonemes['coda'].index(phon[k:k+2])] = 1

        return onset + vowel + codas

    def get_graphemes(self, word):
        word = str(word).upper()
        if word == "NAN":
            word = "NULL"

        # set initial grapheme vectors to zero
        onset = [0 for i in range(len(self.graphemes['onset']))]
        vowel = [0 for i in range(len(self.graphemes['vowel']))]
        codas = [0 for i in range(len(self.graphemes['coda']))]

        # for onset
        for i in range(len(word)):
            if word[i] in self.graphemes['vowel']:
                break
            if word[i] in self.graphemes['onset']:
                onset[self.graphemes['onset'].index(word[i])] = 1
            if word[i:i+2] in self.graphemes['onset']:
                onset[self.graphemes['onset'].index(word[i:i+2])] = 1

        vowel[self.graphemes['vowel'].index(word[i])] = 1

        if i + 1 < len(word):
            if word[i+1] in self.graphemes['vowel']:
                vowel[self.graphemes['vowel'].index(word[i+1])] = 1
            if word[i:i+2] in self.graphemes['vowel']:
                vowel[self.graphemes['vowel'].index(word[i:i+2])] = 1
            if word[i+1] in self.graphemes['vowel'] or word[i:i+2] in self.graphemes['vowel']:
                i += 1

        for j in range(i+1, len(word)):
            if word[j] in self.graphemes['coda']:
                codas[self.graphemes['coda'].index(word[j])] = 1
            if word[j:j+2] in self.graphemes['coda']:
                codas[self.graphemes['coda'].index(word[j:j+2])] = 1
            if word[j:j+3] in self.graphemes['coda']:
                codas[self.graphemes['coda'].index(word[j:j+3])] = 1

        return onset + vowel + codas