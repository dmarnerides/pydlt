.. image:: https://readthedocs.org/projects/pydlt/badge/?version=latest
    :target: http://pydlt.readthedocs.io/en/latest/?badge=latest
    :alt: Documentation Status
                

PyTorch Deep Learning Toolbox
=====================================

PyDLT is a set of tools aimed to make experimenting with PyTorch_ easier 
(than it already is).

.. _PyTorch: http://pytorch.org/

Documentation is available here_.

.. _here: http://pydlt.readthedocs.io/

Features
-----------------

- **Trainers** (currently Vanilla, VanillaGAN, WGAN-GP, BEGAN, FisherGAN)

```python
trainer = dlt.train.VanillaGANTrainer(generator, discriminator, g_optim, d_optim)
for batch, (prediction, losses) in trainer(data_loader):
    # Training happens in the iterator and relevant results are returned for each step
```
- Built in configurable **parser** with arguments.

```python
opt = dlt.config.parse() # Has built in options (can add extra)
print('Some Settings: ', opt.experiment_name, opt.batch_size, opt.lr)
```

- **Configuration files** support and parser compatible functions.

.. code:: bash
```bash
$ python main.py @settings.cfg
Some Settings:  config_test 32 0.0001
```

- **HDR imaging** support (.hdr, .exr, and .pfm formats)

```python
img = dlt.hdr.imread('test.pfm')
dlt.hdr.imwrite('test.exr', img)
```

- **Checkpointing** of (torch serializable) objects; Network state dicts supported.

```python
data_chkp = Checkpointer('data')
data_chkp.save(np.array([1,2,3]))
a = data_chkp.load()
```

- **Image operations** and easy conversions between multiple library views (torch, cv, plt)

```python
img = cv2.imread('image.jpg') # Height x Width x Channels - BGR
dlt.viz.imshow(img, view='cv')  # Height x Width x Channels - RGB
tensor_with_torch_view = cv2torch(img) # Channels x Height x Width - RGB
```

- Easy **visualization** (and make_grid supporting Arrays, Tensors, Variables and lists)

```python
for batch, (prediction, loss) in trainer(loader):
    grid = dlt.util.make_grid([ batch[0], batch[1], prediction], size(3, opt.batch_size))
    dlt.viz.imshow(grid, pause=0.01, title='Training Progress')
```

- Model parameter and layer input/outputs/gradients visualization.

```python
net = nn.Sequential(nn.Linear(10, 10))
dlt.viz.modules.forward_hook(net, [nn.Linear], tag='layer_outputs', histogram=False)
net(Variable(torch.Tensor(3,10)))
```

- CSV **Logger**.

```python
log = dlt.util.Logger('losses', ['train_loss', 'val_loss'])
log({'train_loss': 10, 'val_loss':20})
```

- Command line tool for easy **plotting** of CSV files (with live updating).

```bash
$ dlt-plot --file losses.csv train_loss val_loss --refresh 5 --loglog True --tail 100
```

- A minimal **Progress bar** (with global on/off switch).

```python
from dlt.util import barit
barit.silent = False # Default is False
for batch in barit(loader, start='Loading'):
    pass
```

Installation
---------------------

Make sure you have PyTorch_ installed. OpenCV is also required:

```bash    
conda install -c menpo opencv
```

conda install (recommended):
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


```bash
conda install -c demetris pydlt
```

From source:
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

```bash
git clone https://github.com/dmarnerides/pydlt.git
cd pydlt
python setup.py install
```

About
--------

I created this toolbox while learning Python and PyTorch, after working with
(Lua) Torch, to help speed up experiment prototyping.

If you notice something is wrong or missing please do a pull request or
open up an issue.


Contact
----------

Demetris Marnerides: dmarnerides@gmail.com