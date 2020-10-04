# PMSP Torch
# Ian Dennis Miller, Brian Lam, Blair Armstrong

import pandas as pd

# do not warn about assignment to a copy of a pandas object
pd.options.mode.chained_assignment = None

class Frequencies:

    def __init__(self, frequency_filename):

        # load word frequencies
        df_freq = pd.read_csv(frequency_filename, header=0)
        self.frequencies = {}
        for index, row in df_freq.iterrows():
            self.frequencies[row['WORD']] = row['KFFRQ']


    def get_frequency(self, word):
        word = str(word).upper()
        if word in self.frequencies:
            return(self.frequencies[word])
        else:
            return(1)
