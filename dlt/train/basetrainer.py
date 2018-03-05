from ..util import barit
from ..util.misc import _torch_version

class BaseTrainer(object):
    """Generic Base trainer object to inherit functionality from."""
    def __init__(self):
        self.training = True
        self.use_gpu = False
        self._models = {}
        self._losses = {}
        # Compatibility patching volatile / torch.no_grad() !
        # Will remove it once 0.4 is released
        self._use_no_grad = _torch_version.minor > 3 and _torch_version.major == 0
    
    def cuda(self):
        """Sets the trainer to GPU mode. 
        
        If flagged, the data is cast to GPU before every iteration after being
        retrieved from the loader.
        """
        self.use_gpu = True

    def cpu(self):
        """Sets the trainer to CPU mode"""
        self.use_gpu = False

    def train(self):
        """Sets the trainer and models to training mode"""
        self.training = True
        for _, m in self._models.items():
            m.train()

    def eval(self):
        """Sets the trainer and models to inference mode"""
        self.training = False
        for _, m in self._models.items():
            m.eval()

    def loss_names(self, training=None):
        """Returns the name(s)/key(s) of the training or validation loss(es).
        
        Args:
            training (bool, optional): If provided then the training or
                validation losses are returned if True or False respectively.
                If not provided the current mode loss is returned.
        """
        condition = self.training if training is None else training
        mode = 'training' if condition else 'validation'
        return self._losses[mode]

    def loss_names_training(self):
        """Returns the name(s)/key(s) of the training loss(es)."""
        return self.loss_names(True)

    def loss_names_validation(self):
        """Returns the name(s)/key(s) of the validation loss(es)."""
        return self.loss_names(False)

    def _cudata(self, data):
        if self.use_gpu:
            if any([isinstance(data, x) for x in [set, list, tuple]]):
                data = type(data)(self._cudata(x) for x in data)
            else:
                data = data.cuda()
        return data

    def iteration(self, data):
        raise NotImplementedError

    def iterate(self, loader):
        """Performs an epoch of training or validation.
        
        Args:
            loader (iterable): The data loader.
        """
        for data in barit(loader, start='Training' if self.training else 'Validation'):
            data = self._cudata(data)
            yield data, self.iteration(data)
        return

    __call__ = iterate