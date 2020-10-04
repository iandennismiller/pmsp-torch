# PMSP Torch
# Ian Dennis Miller, Brian Lam, Blair Armstrong

import pandas as pd

# do not warn about assignment to a copy of a pandas object
pd.options.mode.chained_assignment = None


class Graphemes:
    def __init__(self):
        self.graphemes = {
            'onset': ['Y', 'S', 'P', 'T', 'K', 'Q', 'C', 'B', 'D', 'G', 'F', 'V', 'J', 'Z', 'L', 'M', 'N', 'R', 'W', 'H', 'CH', 'GH', 'GN', 'PH', 'PS', 'RH', 'SH', 'TH', 'TS', 'WH'],
            'vowel': ['E', 'I', 'O', 'U', 'A', 'Y', 'AI', 'AU', 'AW', 'AY', 'EA', 'EE', 'EI', 'EU', 'EW', 'EY', 'IE', 'OA', 'OE', 'OI', 'OO', 'OU', 'OW', 'OY', 'UE', 'UI', 'UY'],
            'coda': ['H', 'R', 'L', 'M', 'N', 'B', 'D', 'G', 'C', 'X', 'F', 'V', 'J', 'S', 'Z', 'P', 'T', 'K', 'Q', 'BB', 'CH', 'CK', 'DD', 'DG', 'FF', 'GG', 'GH', 'GN', 'KS', 'LL', 'NG', 'NN', 'PH', 'PP', 'PS', 'RR', 'SH', 'SL', 'SS', 'TCH', 'TH', 'TS', 'TT', 'ZZ', 'U', 'E', 'ES', 'ED'],
        }

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
