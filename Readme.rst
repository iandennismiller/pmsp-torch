PMSP Torch
==========

Installation
------------

::

    git clone https://projects.sisrlab.com/cap-lab/pmsp-torch
    make requirements install

Direct from git using pip
^^^^^^^^^^^^^^^^^^^^^^^^^

An alternative is to use pip, which will install requirements and put the `pmsp.py` script in the PATH.

::

    pip install git+https://projects.sisrlab.com/cap-lab/pmsp-torch@master

Running simulations
-------------------

`pmsp-torch` can be used to replicate several studies based on PMSP 1996.

::

    make pmsp-1996

The experiment can be launched with `bin/pmsp.py`, which takes command line arguments and handles logging.

::

    bin/pmsp.py pmsp-1996 --no-retrain

The experiment can also be launched directly, albeit without arguments:

::

    pmsp_experiments/pmsp_1996.py

Example Notebooks
-----------------

`pmsp-torch` runs inside Jupyter notebooks.

Jupyter:

https://projects.sisrlab.com/cap-lab/pmsp-torch/-/blob/master/pmsp-1996-example.ipynb

Google Colab:

https://colab.research.google.com/drive/1U9htm2E6Eqsz-RLUjy8mXVhkgr_t-Lqv

Windows Torch Installation
--------------------------

::

    pip3 install torch===1.3.1 -f https://download.pytorch.org/whl/torch_stable.html
