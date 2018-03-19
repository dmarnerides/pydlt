from torch.utils.data import DataLoader
from ..util import barit, DirectoryDataset, LoadedDataset
from ..hdr.io import imread
from .opts import fetch_opts

def _custom_get_item(self, index):
    if self.train:
        img, target = self.train_data[index], self.train_labels[index]
    else:
        img, target = self.test_data[index], self.test_labels[index]
    if self.transform is not None:
        img = self.transform(img)
    if self.target_transform is not None:
        target = self.target_transform(target)
    return img, target

def torchvision_dataset(transform=None, train=True, subset=None):
    """Creates a dataset from torchvision, configured using Command Line Arguments.

    Args:
        transform (callable, optional): A function that transforms an image (default None).
        train (bool, optional): Training set or validation - if applicable (default True).
        subset (string, optional): Specifies the subset of the relevant
            categories, if any of them was split (default, None).

    Relevant Command Line Arguments:

        - **dataset**: `--data`, `--torchvision_dataset`.

    Note:
        Settings are automatically acquired from a call to :func:`dlt.config.parse`
        from the built-in ones. If :func:`dlt.config.parse` was not called in the 
        main script, this function will call it.

    Warning:
        Unlike the torchvision datasets, this function returns a dataset that
        uses NumPy Arrays instead of a PIL Images.
    """
    opts = fetch_opts(['dataset'], subset)

    if opts.torchvision_dataset is None:
        if subset is not None:
            apnd = '_' + subset
        else:
            apnd = ''
        raise ValueError('No value given for --torchvision_dataset{0}.'.format(apnd))

    if opts.torchvision_dataset == 'mnist':
        from torchvision.datasets import MNIST
        MNIST.__getitem__ = _custom_get_item
        ret_dataset = MNIST(opts.data, train=train, download=True, transform=transform)
        # Add channel dimension and make numpy for consistency
        if train:
            ret_dataset.train_data = ret_dataset.train_data.unsqueeze(3).numpy()
            ret_dataset.train_labels = ret_dataset.train_labels.numpy()
        else:
            ret_dataset.test_data = ret_dataset.test_data.unsqueeze(3).numpy()
            ret_dataset.test_labels = ret_dataset.test_labels.numpy()
    elif opts.torchvision_dataset == 'fashionmnist':
        from torchvision.datasets import FashionMNIST
        FashionMNIST.__getitem__ = _custom_get_item
        ret_dataset = FashionMNIST(opts.data, train=train, download=True, transform=transform)
        if train:
            ret_dataset.train_data = ret_dataset.train_data.unsqueeze(3).numpy()
            ret_dataset.train_labels = ret_dataset.train_labels.numpy()
        else:
            ret_dataset.test_data = ret_dataset.test_data.unsqueeze(3).numpy()
            ret_dataset.test_labels = ret_dataset.test_labels.numpy()
    elif opts.torchvision_dataset == 'cifar10':
        from torchvision.datasets import CIFAR10
        CIFAR10.__getitem__ = _custom_get_item
        ret_dataset = CIFAR10(opts.data, train=train, download=True, transform=transform)
    elif opts.torchvision_dataset == 'cifar100':
        from torchvision.datasets import CIFAR100
        CIFAR100.__getitem__ = _custom_get_item
        ret_dataset = CIFAR100(opts.data, train=train, download=True, transform=transform)
    return ret_dataset

def directory_dataset(load_fn=imread, preprocess=None, subset=None):
    """Creates a :class:`dlt.util.DirectoryDataset`, configured using Command Line Arguments.

    Args:
        load_fn (callable, optional): Function that loads the data files
            (default :func:`dlt.hdr.imread`).
        preprocess (callable, optional): A function that takes a single data
            point from the dataset to preprocess on the fly (default None).
        subset (string, optional): Specifies the subset of the relevant
            categories, if any of them was split (default, None).

    Relevant Command Line Arguments:

        - **dataset**: `--data`, `--load_all`, `--extensions`.
        - **dataloader**: `--num_threads`.

    Note:
        Settings are automatically acquired from a call to :func:`dlt.config.parse`
        from the built-in ones. If :func:`dlt.config.parse` was not called in the 
        main script, this function will call it.

    """
    opts = fetch_opts(['dataset', 'dataloader'], subset)
    if opts.load_all:
        dummy_set = DirectoryDataset(opts.data, extensions=opts.extensions, load_fn=load_fn)
        dummy_loader = DataLoader(dummy_set, batch_size=1, num_workers=opts.num_threads, pin_memory=False)
        loaded_set = [batch[0].clone().numpy() for batch in barit(dummy_loader, start='Loading')]
        ret_dataset = LoadedDataset(loaded_set, preprocess=preprocess)
        print('Done loading from {0}'.format(opts.data))
    else:
        ret_dataset = DirectoryDataset(opts.data, load_fn=load_fn, preprocess=preprocess, extensions=opts.extensions)
        print('Created dataset from {0}'.format(opts.data))
    return ret_dataset


def loader(dataset, preprocess=None, subset=None):
    """Creates a torch DataLoader using the dataset, configured using Command Line Arguments.

    Args:
        dataset (Dataset): A torch compatible dataset.
        preprocess (callable, optional): A function that takes a single data
            point from the dataset to preprocess on the fly (default None).
        subset (string, optional): Specifies the subset of the relevant
            categories, if any of them was split (default, None).

    Relevant Command Line Arguments:
        
        - **dataloader**: `--batch_size`, `--num_threads`, `--pin_memory`,
          `--shuffle`, `--drop_last`.

    Note:
        Settings are automatically acquired from a call to :func:`dlt.config.parse`
        from the built-in ones. If :func:`dlt.config.parse` was not called in the 
        main script, this function will call it.

    """
    opts = fetch_opts(['dataloader'], subset)
    return DataLoader(LoadedDataset(dataset,preprocess),
                      batch_size=opts.batch_size, num_workers=opts.num_threads,
                      pin_memory=opts.pin_memory, shuffle=opts.shuffle,
                      drop_last=opts.drop_last)
