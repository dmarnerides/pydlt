Full Example
===========================================

The following example implements a configurable training of GANs.
It includes multiple GAN training types (Vanilla, WGAN-GP, FisherGAN, BEGAN)
and multiple datasets (MNIST, FashionMNIST, CIFAR10/100). It can be
extended relatively straightforwardly.

The code consists of two files, *main.py* and *models.py* along with a
configuration file *settings.cfg*. To run:

.. code-block:: bash

    $ python main.py @settings.cfg

It is worth noting:

    - The networks are saved every epoch. Restarting continues from the
      previous checkpoint.
    - Changing --experiment_save creates a new directory for the experiment
      with all the checkpoints and log.

**main.py**:

.. literalinclude:: ../../../examples/gans/main.py

**models.py**:

.. literalinclude:: ../../../examples/gans/models.py

**settings.cfg**:

.. literalinclude:: ../../../examples/gans/settings.cfg