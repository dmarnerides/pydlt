import os
from torch.utils.data import Dataset
from .paths import process

# Helper class for pre-processing loaded datasets
class LoadedDataset(Dataset):
    """Create a torch Dataset from data in memory with on the fly pre-processing.

    Useful when to use with torch DataLoader.

    Args:
        dataset (sequence or collection): A sequence or collection of data points
            that can be indexed.
        preprocess (callable, optional): A function that takes a single data
            point from the dataset to preprocess on the fly (default None).

    Example:
        >>> a = [1.0, 2.0, 3.0]
        >>> a_dataset = dlt.util.LoadedDataset(a, lambda x: x**2)
        >>> loader = torch.utils.data.DataLoader(a_dataset, batch_size=3)
        >>> for val in loader:
        >>>     print(val)
        1
        4
        9
        [torch.DoubleTensor of size 3]

    """
    def __init__(self, dataset, preprocess=None):
        super(LoadedDataset, self).__init__()
        self.dataset = dataset
        if preprocess is None:
            preprocess = lambda x: x
        self.preprocess = preprocess
    def __getitem__(self, index):
        return self.preprocess(self.dataset[index])
    def __len__(self):
        return len(self.dataset)

class DirectoryDataset(Dataset):
    """Creates a dataset recursively (no structure requirement).
    
    Similar to `torchvision.datasets.FolderDataset`, however there is no need for
    a specific directory structure, or data format.

    Args:
        data_root (string): Path to root directory of data.
        extensions (list or tuple): Extensions/ending patterns of data files.
        loader (callable): Function that loads the data files.
        preprocess (callable, optional): A function that takes a single data
            point from the dataset to preprocess on the fly (default None).

    """
    def __init__(self, data_root, extensions, load_fn, preprocess=None):
        super(DirectoryDataset, self).__init__()
        data_root = process(data_root)
        self.file_list = []
        for root, _, fnames in sorted(os.walk(data_root)):
            for fname in fnames:
                if any(fname.lower().endswith(extension) for extension in extensions):
                    self.file_list.append(os.path.join(root, fname))
        if len(self.file_list) == 0:
            msg = 'Could not find any files with extensions:\n[{0}]\nin\n{1}'
            raise RuntimeError(msg.format(', '.join(extensions),data_root))

        self.preprocess = preprocess
        self.load_fn = load_fn

    def __getitem__(self, index):
        dpoint = self.load_fn(self.file_list[index])
        if self.preprocess is not None:
            dpoint = self.preprocess(dpoint)
        return dpoint

    def __len__(self):
        return len(self.file_list)