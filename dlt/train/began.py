import math
import torch
from torch.autograd import Variable
from .ganbasetrainer import GANBaseTrainer
from ..util.misc import _get_scalar_value

class BEGANTrainer(GANBaseTrainer):
    """Boundary Equilibrium GAN trainer. 
        
    Args:
        generator (nn.Module): The generator network.
        discriminator (nn.Module): The discriminator network.
        g_optimizer (torch.optim.Optimizer): Generator Optimizer.
        d_optimizer (torch.optim.Optimizer): Discriminator Optimizer.
        lambda_k (float): Learning rate of k parameter.
        gamma (float): Diversity ratio.
        d_iter (int): Number of discriminator steps per generator step.
        add_loss (callable, optional): Extra loss term to be added to GAN
            objective.

    Each iteration returns the mini-batch and a tuple containing:

        - The generator prediction.
        - A dictionary containing a `d_loss` (not when validating) and a 
          `g_loss` dictionary (only if a generator step is performed):
            
            - `d_loss contains`: `d_loss`, `real_loss`, `fake_loss`, `k`,
              `balance`, and `measure`.
            - `g_loss` contains: `g_loss` (and extra_loss if add_loss is used).
    
    Example:

    .. code-block:: python3
    
        >>> trainer = dlt.train.BEGANTrainer(gen, disc, g_optim, d_optim, lambda_k, gamma)
        >>> # Training mode
        >>> trainer.train()
        >>> for batch, (prediction, loss) in trainer(train_data_loader):
        >>>     print(loss['d_loss']['measure'])
    """
    def __init__(self, generator, discriminator, g_optimizer, d_optimizer, lambda_k, gamma, d_iter=1, add_loss=None):
        super(BEGANTrainer, self).__init__(generator, discriminator, g_optimizer, 
                                                d_optimizer, d_iter, add_loss)
        # Register losses
        self._losses['training'] = ['d_loss', 'real_loss', 'fake_loss', 'k', 'balance', 'measure']
        self._losses['validation'] = ['g_loss']
        self.k = 0.0
        self.lambda_k = lambda_k
        self.gamma = gamma
        if self.add_loss is not None:
            self._losses['training'] += ['extra_loss']

    def d_step(self, g_input, real_input):
        for p in self._models['discriminator'].parameters():
            p.requires_grad = True
        self._models['discriminator'].zero_grad()
        if self._use_no_grad:
            with torch.no_grad():
                t_pred = self._models['generator'](Variable(g_input)).data
            prediction = Variable(t_pred)
        else:
            prediction = Variable(self._models['generator'](Variable(g_input, volatile=True)).data)
        fake_loss = (self._models['discriminator'](prediction) - prediction).abs().mean()
        v_real_input = Variable(real_input)
        real_loss = (self._models['discriminator'](v_real_input) - v_real_input).abs().mean()
        
        d_loss = real_loss - self.k*fake_loss
        
        d_loss.backward()
        self._optimizers['discriminator'].step()

        balance = (self.gamma * _get_scalar_value(real_loss.data) - _get_scalar_value(fake_loss.data))
        self.k = min(max(self.k + self.lambda_k*balance, 0), 1)
        measure = _get_scalar_value(real_loss.data) + math.fabs(balance)

        ret_losses = {'d_loss': _get_scalar_value(d_loss.data),
                      'real_loss': _get_scalar_value(real_loss.data),
                      'fake_loss': _get_scalar_value(fake_loss.data),
                      'k': self.k, 'measure': measure, 'balance': balance}
        self.d_iter_counter += 1
        return prediction.data, ret_losses

    def g_step(self, g_input, real_input):
        for p in self._models['discriminator'].parameters():
            p.requires_grad = False
        if self.training:
            self._models['generator'].zero_grad()
            prediction = self._models['generator'](Variable(g_input))
            error = (self._models['discriminator'](prediction) - prediction).abs().mean()
            total_loss = error
            if self.add_loss:
                extra_loss = self.add_loss(prediction, Variable(real_input))
                total_loss += extra_loss
            total_loss.backward()
            self._optimizers['generator'].step()
        else:
            if self._use_no_grad:
                with torch.no_grad():
                    prediction = self._models['generator'](Variable(g_input))
                    error = (self._models['discriminator'](prediction) - prediction).abs().mean()
                    total_loss = error
                    if self.add_loss:
                        extra_loss = self.add_loss(prediction, Variable(real_input))
                        total_loss += extra_loss
            else:
                prediction = self._models['generator'](Variable(g_input, volatile=True))
                error = (self._models['discriminator'](prediction) - prediction).abs().mean()
                total_loss = error
                if self.add_loss:
                    extra_loss = self.add_loss(prediction, Variable(real_input))
                    total_loss += extra_loss
        ret_loss = {'g_loss': _get_scalar_value(error.data)}
        if self.add_loss:
            ret_loss['extra_loss'] = _get_scalar_value(extra_loss.data)
        return prediction.data, ret_loss
