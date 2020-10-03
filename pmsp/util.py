# PMSP Torch
# CAP Lab

import os
import datetime
import matplotlib.pyplot as plt

def make_losses_figure(losses):
    plt.figure()
    plt.title("Training Curve")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.plot(losses, label="Training Loss")
    return plt

def write_losses(losses, folder):
    plt = make_losses_figure(losses)
    plt.savefig(folder+"/lossplot_final.png", dpi=150)
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
            rootdir = path+"/var/results/"+date+"_test"+'{:02d}'.format(i)
            os.mkdir(rootdir)
            break
        except:
            i += 1

    return(rootdir)
