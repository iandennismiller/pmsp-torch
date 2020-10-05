# PMSP Torch
# Ian Dennis Miller, Brian Lam, Blair Armstrong

import numpy as np
import pandas as pd

# do not warn about assignment to a copy of a pandas object
pd.options.mode.chained_assignment = None


class Phonemes:
    def __init__(self):
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

    def one_hot_to_phoneme(self, bank_str, index):
        return self.phonemes[bank_str][index]

    def expand_one_hot(self, vector):
        np_vector = np.array(vector)

        onset_idx = np_vector[:22].argmax()
        vowel_idx = np_vector[23:37].argmax()
        offset_idx = np_vector[38:].argmax()

        return self.one_hot_to_phoneme('onset', onset_idx) + \
            self.one_hot_to_phoneme('vowel', vowel_idx) + \
            self.one_hot_to_phoneme('coda', offset_idx)
