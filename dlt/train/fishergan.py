import torch
from torch.autograd import Variable
from .ganbasetrainer import GANBaseTrainer
from ..util.misc import _get_scalar_value

class FisherGANTrainer(GANBaseTrainer):
    """Fisher GAN trainer. 
    
    Args:
        generator (nn.Module): The generator network.
        discriminator (nn.Module): The discriminator network.
        g_optimizer (torch.optim.Optimizer): Generator Optimizer.
        d_optimizer (torch.optim.Optimizer): Discriminator Optimizer.
        rho (float): Quadratic penalty weight.
        d_iter (int, optional): Number of discriminator steps per generator
            step (default 1).
        add_loss (callable, optional): Extra loss term to be added to GAN
            objective (default None).

    Each iteration returns the mini-batch and a tuple containing:

        - The generator prediction.
        - A dictionary containing a `d_loss` (not when validating) and a 
          `g_loss` dictionary (only if a generator step is performed):
            
            - `d_loss contains`: `ipm_enum`, `ipm_denom`, `ipm_ratio`, 
              `d_loss`, `constraint`, `epf`, `eqf`, `epf2`, `eqf2` and
              `lagrange`.
            - `g_loss` contains: `g_loss` (and extra_loss if add_loss is used).

    Example:
        >>> trainer = dlt.train.FisherGANTrainer(gen, disc, g_optim, d_optim, rho)
        >>> # Training mode
        >>> trainer.train()
        >>> for batch, (prediction, loss) in trainer(train_data_loader):
        >>>     print(loss['d_loss']['constraint'])
    """
    def __init__(self, generator, discriminator, g_optimizer, d_optimizer, rho, d_iter=1, add_loss=None):
        super(FisherGANTrainer, self).__init__(generator, discriminator, g_optimizer, 
                                                d_optimizer, d_iter, add_loss)
        # Register losses
        self._losses['training'] = ['ipm_enum', 'ipm_denom', 'ipm_ratio', 'd_loss', 'constraint', 
                                    'epf', 'eqf', 'epf2', 'eqf2', 'lagrange']
        self._losses['validation'] = ['g_loss']
        self.rho = rho
        self.alpha = None
        if self.add_loss is not None:
            self._losses['training'] += ['extra_loss']

    def d_step(self, g_input, real_input):
        for p in self._models['discriminator'].parameters():
            p.requires_grad = True
        self._models['discriminator'].zero_grad()
        if self.alpha is None:
            self.alpha = Variable(g_input.new([0]), requires_grad=True)
        
        if self._use_no_grad:
            with torch.no_grad():
                t_pred = self._models['generator'](Variable(g_input)).data
            prediction = Variable(t_pred)
        else:
            prediction = Variable(self._models['generator'](Variable(g_input, volatile=True)).data)
        vphi_fake = self._models['discriminator'](prediction)
        vphi_real = self._models['discriminator'](Variable(real_input))

        epf, eqf = vphi_real.mean(), vphi_fake.mean()
        epf2, eqf2 = (vphi_real**2).mean(), (vphi_fake**2).mean()
        constraint = (1- (0.5*epf2 + 0.5*eqf2))
        d_loss = -(epf - eqf + self.alpha*constraint - self.rho/2 * constraint**2)
        
        d_loss.backward()
        self._optimizers['discriminator'].step()
        self.alpha.data += self.rho * self.alpha.grad.data
        self.alpha.grad.data.zero_()

        # IPM
        ipm_enum = _get_scalar_value(epf.data) - _get_scalar_value(eqf.data)
        ipm_denom = (0.5*_get_scalar_value(epf2.data) + 0.5*_get_scalar_value(eqf2.data))**0.5
        ipm_ratio = ipm_enum/ipm_denom
        ret_losses = {'ipm_enum': ipm_enum, 'ipm_denom': ipm_denom, 
                      'ipm_ratio': ipm_ratio, 
                      'd_loss': -_get_scalar_value(d_loss.data), 
                      'constraint': 1 - _get_scalar_value(constraint.data),
                      'epf': _get_scalar_value(epf.data),
                      'eqf': _get_scalar_value(eqf.data),
                      'epf2': _get_scalar_value(epf2.data),
                      'eqf2': _get_scalar_value(eqf2.data),
                      'lagrange': _get_scalar_value(self.alpha.data)}
        self.d_iter_counter += 1
        return prediction.data, ret_losses

    def g_step(self, g_input, real_input):
        for p in self._models['discriminator'].parameters():
            p.requires_grad = False
        if self.training:
            self._models['generator'].zero_grad()
            prediction = self._models['generator'](Variable(g_input))
            error = - self._models['discriminator'](prediction).mean()
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
                    error = - self._models['discriminator'](prediction).mean()
                    total_loss = error
                    if self.add_loss:
                        extra_loss = self.add_loss(prediction, Variable(real_input))
                        total_loss += extra_loss
            else:
                prediction = self._models['generator'](Variable(g_input, volatile=True))
                error = - self._models['discriminator'](prediction).mean()
                total_loss = error
                if self.add_loss:
                    extra_loss = self.add_loss(prediction, Variable(real_input))
                    total_loss += extra_loss
        ret_loss = {'g_loss': _get_scalar_value(error.data)}
        if self.add_loss:
            ret_loss['extra_loss'] = _get_scalar_value(extra_loss.data)
        return prediction.data, ret_loss