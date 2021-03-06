import torch
from ..util import barit

class BaseTrainer(object):
    """Generic Base trainer object to inherit functionality from."""
    def __init__(self):
        self.training = True
        self.device = 'cpu'
        self._models = {}
        self._optimizers = {}
        self._losses = {}
        self.epoch = 1
    
    def cuda(self, device=0):
        """Sets the trainer to GPU mode. 
        
        If flagged, the data is cast to GPU before every iteration after being
        retrieved from the loader.
        """
        self.device = 'cuda:{0}'.format(device)

    def cpu(self, device=0):
        """Sets the trainer to CPU mode"""
        self.device = 'cpu:{0}'.format(device)

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

    def _to(self, data):
        if any([isinstance(data, x) for x in [set, list, tuple]]):
            return type(data)(self._to(x) for x in data)
        else:
            return data.detach().to(self.device)

    def iteration(self, data):
        raise NotImplementedError

    def iterate(self, loader):
        """Performs an epoch of training or validation.
        
        Args:
            loader (iterable): The data loader.
        """
        
        torch.set_grad_enabled(self.training)

        for data in barit(loader, start='Training' if self.training else 'Validation'):
            data = self._to(data)
            yield data, self.iteration(data)
        
        if self.training:
            self.epoch += 1

        return

    __call__ = iterate

    def __getstate__(self):
        return self.state_dict()

    def __setstate__(self, state):
        self.load_state_dict(state)

    def state_dict(self):
        """Returns the state of the trainer as a :class:`dict`.

        It contains an entry for every variable in self.__dict__ which
        is not the one of the models or optimizers.
        """
        return {key: value for key, value in self.__dict__.items() 
                if key not in ['_optimizers', '_losses', '_models']}

    def load_state_dict(self, state_dict):
        """Loads the trainers state.

        Arguments:
            state_dict (dict): scheduler state. Should be an object returned
                from a call to :meth:`state_dict`.
        """
        self.__dict__.update(state_dict)