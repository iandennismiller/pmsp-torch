# PMSP Torch
# CAP Lab

import os
import datetime
import matplotlib.pyplot as plt

def write_losses(losses, folder):
    plt.figure()
    plt.title("Training Curve")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.plot(losses, label="Training Loss")
    plt.savefig(folder+"/lossplot_final.png", dpi=150)
    plt.close()

def make_folder():
    # create a new folder for each run
    path = os.getcwd()
    now = datetime.datetime.now()
    date = now.strftime("%b").lower()+now.strftime("%d")

    if not os.path.isdir(os.path.join(path, 'build')):
        os.mkdir(os.path.join(path, 'build'))

    i = 1
    while True:
        try:
            rootdir = path+"/build/"+date+"_test"+'{:02d}'.format(i)
            os.mkdir(rootdir)
            break
        except:
            i += 1

    return(rootdir)
