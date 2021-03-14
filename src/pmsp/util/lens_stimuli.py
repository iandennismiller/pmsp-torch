# PMSP Torch
# Ian Dennis Miller, Brian Lam, Blair Armstrong

import re
import random

def generate_stimuli(df):
    result = ""
    count = 0

    for idx in range(0, len(df)):
        item = df.iloc[idx, : ]
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

def generate_stimuli_log_transform(df, percentage=0.5):
    buffer = {
        'training': "",
        'testing': "",
    }

    full_set = set(range(0, len(df)))
    training_set = set(random.sample(full_set, int(len(df) * percentage)))
    testing_set = full_set.difference(training_set)

    sets = [
        ('training', training_set),
        ('testing', testing_set)
    ]

    for current_buffer, current_set in sets:
        count = 0
        for idx in current_set:
            item = df.iloc[idx, : ]
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

def generate_stimuli_the_normalized(df):
    """
    Normalize frequencies relative to the word THE, with a frequency of 69971
    """
    result = ""
    count = 0

    for idx in range(0, len(df)):
        item = df.iloc[idx, : ]
        item['phon'] = re.sub(r'\W', '', item['phon'])
        if item['type'] == '#': item['type'] = '0'
        result += "name: {{{count}_{orth}_{phon}_{type}}}\n".format(count=count, **item)

        result += "freq: {0:.9f}\n".format(round(item['frequency']/69971.0, 8))

        in_vec = ' '.join(str(x) for x in item['graphemes'])
        result += "I: {}\n".format(in_vec)

        target_vec = ' '.join(str(x) for x in item['phonemes'])
        result += "T: {};\n\n".format(target_vec)
        count += 1

    return(result)
