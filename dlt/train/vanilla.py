import torch
from torch.autograd import Variable
from .basetrainer import BaseTrainer
from ..util.misc import _get_scalar_value

class VanillaTrainer(BaseTrainer):
    """Training of a network using a criterion/loss function.
    
    Args:
        model (nn.Module): The network to train.
        criterion (callable): The function to optimize.
        optimizer (torch.optim.Optimizer): A torch Optimizer.

    Each iteration returns the mini-batch and a tuple containing:

        - The model prediction.
        - A dictionary with the `training_loss` or `validation_loss`.
    
    Example:
        >>> trainer = dlt.train.VanillaTrainer(my_model, nn.L1Loss(), my_optimizer)
        >>> # Training mode
        >>> trainer.train()
        >>> for batch, (prediction, loss) in trainer(train_data_loader):
        >>>     print(loss['training_loss'])
        >>> # Validation mode
        >>> trainer.eval()
        >>> for batch, (prediction, loss) in trainer(valid_data_loader):
        >>>     print(loss['validation_loss'])
    """
    def __init__(self, model, criterion, optimizer):
        super(VanillaTrainer, self).__init__()
        # Register models and losses
        self._models['model'] = model
        self._optimizers['optimizer'] = optimizer
        
        self._losses['training'] = ['training_loss']
        self._losses['validation'] = ['validation_loss']

        self.criterion = criterion

    def iteration(self, data):

        if self.training:
            v_input = Variable(data[0])
            v_output = Variable(data[1])
            v_pred = self._models['model'](v_input)
            loss = self.criterion(v_pred, v_output)
            self._optimizers['optimizer'].zero_grad()
            loss.backward()
            self._optimizers['optimizer'].step()
        else:
            if self._use_no_grad:
                with torch.no_grad():
                    v_input = Variable(data[0])
                    v_output = Variable(data[1])
                    v_pred = self._models['model'](v_input)
                    loss = self.criterion(v_pred, v_output)
            else:
                v_input = Variable(data[0], volatile=True)
                v_output = Variable(data[1], volatile=True)
                v_pred = self._models['model'](v_input)
                loss = self.criterion(v_pred, v_output)
        key = 'training_loss' if self.training else 'validation_loss'
        return v_pred.data, {key: _get_scalar_value(loss.data)}
