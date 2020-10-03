# PMSP Torch
# Ian Dennis Miller, Brian Lam, Blair Armstrong

import os
import datetime
import matplotlib.pyplot as plt


def write_figure(dataseries, title, xlabel, ylabel, filename):
    plt.figure()
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.plot(dataseries, label=title)
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
