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

    make replicate-adkp-2017

Build LENS stimuli
------------------

::

    make lens-stimuli

Code Example
------------

This works in Jupyter or Google Colab:

https://colab.research.google.com

::

    from PMSP.stimuli import PMSPStimuli
    from PMSP.simulator import Simulator
    from PMSP.util import make_losses_figure

    sim = Simulator()
    sim.train(num_epochs=400)
    make_losses_figure(sim.losses)

Windows Torch Installation
--------------------------

::

    pip3 install torch===1.3.1 -f https://download.pytorch.org/whl/torch_stable.html
