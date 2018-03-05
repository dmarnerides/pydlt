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
        - A dictionary with:
            
            - `d_loss`, `real_loss`, `fake_loss`, `k`, `balance`, `measure`
                if training mode.
            - `g_loss` if validation mode.
    
    Example:

    .. code-block:: python3
    
        >>> trainer = dlt.train.BEGANTrainer(gen, disc, g_optim, d_optim, lambda_k, gamma)
        >>> # Training mode
        >>> trainer.train()
        >>> for batch, (prediction, loss) in trainer(train_data_loader):
        >>>     print(loss['measure'])
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

    def d_step(self, g_input, real_input):
        for p in self.discriminator.parameters():
            p.requires_grad = True
        self.discriminator.zero_grad()
        if self._use_no_grad:
            with torch.no_grad():
                t_pred = self.generator(Variable(g_input)).data
            prediction = Variable(t_pred)
        else:
            prediction = Variable(self.generator(Variable(g_input, volatile=True)).data)
        fake_loss = (self.discriminator(prediction) - prediction).abs().mean()
        v_real_input = Variable(real_input)
        real_loss = (self.discriminator(v_real_input) - v_real_input).abs().mean()
        
        d_loss = real_loss - self.k*fake_loss

        if self.add_loss:
            d_loss = d_loss + self.add_loss(prediction, Variable(real_input))
        
        d_loss.backward()
        self.d_optimizer.step()

        balance = (self.gamma * _get_scalar_value(real_loss.data) - _get_scalar_value(fake_loss.data))
        self.k = min(max(self.k + self.lambda_k*balance, 0), 1)
        measure = _get_scalar_value(real_loss.data) + math.fabs(balance)

        ret_losses = {'d_loss': _get_scalar_value(d_loss.data),
                      'real_loss': _get_scalar_value(real_loss.data),
                      'fake_loss': _get_scalar_value(fake_loss.data),
                      'k': self.k, 'measure': measure, 'balance': balance}
        self.d_iter_counter += 1
        return prediction.data, ret_losses

    def g_step(self, g_input):
        for p in self.discriminator.parameters():
            p.requires_grad = False
        if self.training:
            self.generator.zero_grad()
            prediction = self.generator(Variable(g_input))
            error = (self.discriminator(prediction) - prediction).abs().mean()
            error.backward()
            self.g_optimizer.step()
        else:
            if self._use_no_grad:
                with torch.no_grad():
                    prediction = self.generator(Variable(g_input))
                    error = (self.discriminator(prediction) - prediction).abs().mean()
            else:
                prediction = self.generator(Variable(g_input, volatile=True))
                error = (self.discriminator(prediction) - prediction).abs().mean()
        return prediction.data, {'g_loss': _get_scalar_value(error.data)}
