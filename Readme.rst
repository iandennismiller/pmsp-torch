PMSP Torch
==========

Installation
------------

::

    git clone https://projects.sisrlab.com/cap-lab/pmsp-torch
    make requirements
    make test

Or install straight from git
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

::

    pip install git+https://projects.sisrlab.com/cap-lab/pmsp-torch@master
    pmsp.py test
    pmsp.py simulate

Running simulation
------------------

::

    make run

Viewing stimuli
---------------

::

    make view

Writing stimuli to a file
-------------------------

::

    make write

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
