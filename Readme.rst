PMSP Torch
==========

Installation
------------

::

    git clone https://projects.sisrlab.com/cap-lab/pmsp-torch
    make requirements

Or install straight from git using pip
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

::

    pip install git+https://projects.sisrlab.com/cap-lab/pmsp-torch@master

Running simulation
------------------

::

    make replicate-pmsp-1996
    make replicate-adkp-2017

Example
-------

This works in Jupyter or Google Colab:

- https://colab.research.google.com
- https://colab.research.google.com/drive/1U9htm2E6Eqsz-RLUjy8mXVhkgr_t-Lqv

Configure colab
^^^^^^^^^^^^^^^

::

    !pip3 -q install --upgrade git+https://projects.sisrlab.com/cap-lab/pmsp-torch@master
    !git clone https://projects.sisrlab.com/cap-lab/pmsp-torch.git
    
    # # Optionally, mount from google drive to persist logs and images
    # from google.colab import drive
    # drive.mount('/content/drive')

Run a sample network
^^^^^^^^^^^^^^^^^^^^

::

    import logging
    import torch
    import torch.optim as optim

    from pmsp.stimuli import build_dataloader
    from pmsp.network import PMSPNetwork
    from pmsp.trainer import PMSPTrainer
    from pmsp.util import plot_figure

    logging.basicConfig(
        # filename='/content/drive/My Drive/Colab Notebooks/pmsp.log',
        level=logging.INFO
    )

    pmsp_stimuli, pmsp_dataset, pmsp_dataloader = build_dataloader(
        mapping_filename="pmsp-torch/pmsp/data/plaut_dataset_collapsed.csv",
        frequency_filename="pmsp-torch/pmsp/data/word-frequencies.csv"
    )

    torch.manual_seed(1)

    network = PMSPNetwork()
    trainer = PMSPTrainer(network=network)

    optimizers = {
        0: optim.SGD(network.parameters(), lr=0.0001),
        10: optim.Adam(network.parameters(), lr=0.01)
    }

    losses = trainer.train(
        dataloader=pmsp_dataloader,
        num_epochs=350,
        optimizers=optimizers
    )

    plot_figure(
        dataseries=losses,
        title="Average Loss over Time",
        xlabel="epoch",
        ylabel="average loss"
    )

Windows Torch Installation
--------------------------

::

    pip3 install torch===1.3.1 -f https://download.pytorch.org/whl/torch_stable.html
