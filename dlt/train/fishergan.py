import torch
from .ganbasetrainer import GANBaseTrainer

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

    Each iteration returns the mini-batch and a tuple containing:

        - The generator prediction.
        - A dictionary containing a `d_loss` (not when validating) and a 
          `g_loss` dictionary (only if a generator step is performed):
            
            - `d_loss contains`: `ipm_enum`, `ipm_denom`, `ipm_ratio`, 
              `d_loss`, `constraint`, `epf`, `eqf`, `epf2`, `eqf2` and
              `lagrange`.
            - `g_loss` contains: `g_loss`.

    Example:
        >>> trainer = dlt.train.FisherGANTrainer(gen, disc, g_optim, d_optim, rho)
        >>> # Training mode
        >>> trainer.train()
        >>> for batch, (prediction, loss) in trainer(train_data_loader):
        >>>     print(loss['d_loss']['constraint'])
    """
    def __init__(self, generator, discriminator, g_optimizer, d_optimizer, rho, d_iter=1):
        super(FisherGANTrainer, self).__init__(generator, discriminator, g_optimizer, 
                                                d_optimizer, d_iter)
        # Register losses
        self._losses['training'] = ['ipm_enum', 'ipm_denom', 'ipm_ratio', 'd_loss', 'constraint', 
                                    'epf', 'eqf', 'epf2', 'eqf2', 'lagrange', 'g_loss']
        self._losses['validation'] = ['g_loss']
        self.rho = rho
        self.alpha = torch.zeros(1, requires_grad=True)

    def d_step(self, g_input, real_input):
        disc, gen = self._models['discriminator'], self._models['generator']
        self._set_gradients('discriminator', True)
        self._set_gradients('generator', False)
        self.alpha.to(g_input.device)
    
        prediction = gen(g_input)
        vphi_fake = disc(prediction)
        vphi_real = disc(real_input)

        epf, eqf = vphi_real.mean(), vphi_fake.mean()
        epf2, eqf2 = (vphi_real**2).mean(), (vphi_fake**2).mean()
        constraint = (1- (0.5*epf2 + 0.5*eqf2))
        d_loss = -(epf - eqf + self.alpha*constraint - self.rho/2 * constraint**2)
        
        disc.zero_grad()
        d_loss.backward()
        self._optimizers['discriminator'].step()
        self.alpha = self.alpha + self.rho * self.alpha.grad
        self.alpha.detach().requires_grad_(True)

        # IPM
        ipm_enum = epf.item() - eqf.item()
        ipm_denom = (0.5*epf2.item() + 0.5*eqf2.item())**0.5
        ipm_ratio = ipm_enum/ipm_denom
        ret_losses = {'ipm_enum': ipm_enum, 'ipm_denom': ipm_denom, 
                      'ipm_ratio': ipm_ratio, 
                      'd_loss': -d_loss.item(), 
                      'constraint': 1 - constraint.item(),
                      'epf': epf.item(),
                      'eqf': eqf.item(),
                      'epf2': epf2.item(),
                      'eqf2': eqf2.item(),
                      'lagrange': self.alpha.item()}
        self.d_iter_counter += 1
        return prediction, ret_losses

    def g_step(self, g_input, real_input):
        disc, gen = self._models['discriminator'], self._models['generator']
        self._set_gradients('discriminator', False)
        self._set_gradients('generator', True)

        prediction = gen(g_input)
        loss = - disc(prediction).mean()

        if self.training:
            gen.zero_grad()
            loss.backward()
            self._optimizers['generator'].step()

        return prediction, {'g_loss': loss.item()}