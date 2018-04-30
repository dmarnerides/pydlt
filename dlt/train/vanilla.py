import torch
from torch.autograd import Variable
from .basetrainer import BaseTrainer

class VanillaTrainer(BaseTrainer):
    """Training of a network using a criterion/loss function.
    
    Args:
        model (nn.Module): The network to train.
        criterion (callable): The function to optimize.
        optimizer (torch.optim.Optimizer): A torch Optimizer.

    Each iteration returns the mini-batch and a tuple containing:

        - The model prediction.
        - A dictionary with the `training_loss` or `validation_loss`
          (along with the partial losses, if criterion returns a dictionary).
    
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

    Note:
        If the criterion returns a dict of (named) losses, then they are added
        together to backpropagate. The total is returned along with all the
        partial losses.
    """
    def __init__(self, model, criterion, optimizer):
        super(VanillaTrainer, self).__init__()
        # Register models and losses
        self._models['model'] = model
        self._optimizers['optimizer'] = optimizer
        
        self._losses['training'] = ['training_loss']
        self._losses['validation'] = ['validation_loss']

        self.criterion = criterion
        if hasattr(criterion, 'loss_list'):
            for loss in criterion.loss_list:
                self._losses['training'] += [loss]
                self._losses['validation'] += [loss]


    def iteration(self, data):
        loss_key = 'training_loss' if self.training else 'validation_loss'
        v_pred = self._models['model'](data[0])
        losses = self.criterion(v_pred, data[1])
        if isinstance(losses, dict):
            total_loss = 0
            for key, val in losses.items():
                total_loss = total_loss + val
                losses[key] = val.item()
            losses[loss_key] = total_loss.item()
        else:
            total_loss = losses
            losses = {loss_key: total_loss.item()}

        if self.training:
            self._optimizers['optimizer'].zero_grad()
            total_loss.backward()
            self._optimizers['optimizer'].step()
        
        return v_pred, losses
