.. image:: https://readthedocs.org/projects/pydlt/badge/?version=latest
    :target: http://pydlt.readthedocs.io/en/latest/?badge=latest
    :alt: Documentation Status
                

Deep Learning Toolbox for PyTorch
=====================================

PyDLT is a set of tools aimed to make experimenting with PyTorch_ easier 
(than it already is).

.. _PyTorch: http://pytorch.org/

Documentation is available here_.

.. _here: http://pydlt.readthedocs.io/

Installation
---------------------

Make sure you have PyTorch_ installed. OpenCV is also required:

.. code:: bash
    
    conda install -c menpo opencv

PyDLT can be installed using conda:

.. code:: bash

    conda install -c demetris pydlt

Or from source:

.. code:: bash

    git clone https://github.com/dmarnerides/pydlt.git
    cd pydlt
    python setup.py install

About
--------

I created this toolbox while learning Python and PyTorch, after working with
(Lua) Torch, to help speed up experiment prototyping.

If you notice something is wrong or missing please do a pull request or
open up an issue.


Contact
----------

Demetris Marnerides: dmarnerides@gmail.com