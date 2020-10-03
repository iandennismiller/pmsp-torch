# PMSP Torch
# Ian Dennis Miller, Brian Lam, Blair Armstrong

import re
import random
import pandas as pd
from math import log

from .language.phonemes import Phonemes
from .language.graphemes import Graphemes
from .language.frequencies import Frequencies

# do not warn about assignment to a copy of a pandas object
pd.options.mode.chained_assignment = None

class PMSPStimuli:

    def __init__(self, mapping_filename, frequency_filename):

        # load graphemes, phonemes, and word frequencies
        self.graphemes = Graphemes()
        self.phonemes = Phonemes()
        self.frequencies = Frequencies(frequency_filename)

        self.df = pd.read_csv(mapping_filename, sep=",")

        # If there is a column called "freq" in the dataset, then use that
        try:
            freqs = self.df["freq"]
            lookup_frequencies = False
        except:
            lookup_frequencies = True

        self.df = self.df[["orth", "phon", "type"]]
        self.df["graphemes"] = self.df["orth"].apply(
            lambda x: self.graphemes.get_graphemes(x)
        )
        self.df["phonemes"] = self.df["phon"].apply(
            lambda x: self.phonemes.get_phonemes(x)
        )

        # assign frequencies
        if lookup_frequencies:
            self.df["frequency"] = self.df["orth"].apply(
                lambda x: self.frequencies.get_frequency(x)
            )
        else:
            self.df["frequency"] = freqs

        # standardize frequency
        # freq_sum = self.df["frequency"].sum()
        # self.df["frequency"] = self.df["frequency"].apply(
        #     lambda x: (x / freq_sum)
        # )

        # log transform frequency
        self.df["frequency"] = self.df["frequency"].apply(
            lambda x: log(x+2)
        )

    def generate_stimuli(self):
        result = ""
        count = 0

        for idx in range(0, len(self.df)):
            item = self.df.iloc[idx, : ]
            item['phon'] = re.sub(r'\W', '', item['phon'])
            if item['type'] == '#': item['type'] = '0'
            result += "name: {{{count}_{orth}_{phon}_{type}}}\n".format(count=count, **item)

            result += "freq: {0:.9f}\n".format(item['frequency'])

            in_vec = ' '.join(str(x) for x in item['graphemes'])
            result += "I: {}\n".format(in_vec)

            target_vec = ' '.join(str(x) for x in item['phonemes'])
            result += "T: {};\n\n".format(target_vec)
            count += 1

        return(result)

    def generate_stimuli_log_transform(self, percentage=0.5):
        buffer = {
            'training': "",
            'testing': "",
        }

        full_set = set(range(0, len(self.df)))
        training_set = set(random.sample(full_set, int(len(self.df) * percentage)))
        testing_set = full_set.difference(training_set)

        sets = [
            ('training', training_set),
            ('testing', testing_set)
        ]

        for current_buffer, current_set in sets:
            count = 0
            for idx in current_set:
                item = self.df.iloc[idx, : ]
                item['phon'] = re.sub(r'\W', '', item['phon'])
                if item['type'] == '#': item['type'] = '0'
                buffer[current_buffer] += "name: {{{count}_{orth}_{phon}_{type}}}\n".format(count=count, **item)

                buffer[current_buffer] += "freq: {0:.9f}\n".format(item['frequency'])

                in_vec = ' '.join(str(x) for x in item['graphemes'])
                buffer[current_buffer] += "I: {}\n".format(in_vec)

                target_vec = ' '.join(str(x) for x in item['phonemes'])
                buffer[current_buffer] += "T: {};\n\n".format(target_vec)
                count += 1

        return(buffer)
