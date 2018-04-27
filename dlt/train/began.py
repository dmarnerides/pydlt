import math
from .ganbasetrainer import GANBaseTrainer

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

    Each iteration returns the mini-batch and a tuple containing:

        - The generator prediction.
        - A dictionary containing a `d_loss` (not when validating) and a 
          `g_loss` dictionary (only if a generator step is performed):
            
            - `d_loss contains`: `d_loss`, `real_loss`, `fake_loss`, `k`,
              `balance`, and `measure`.
            - `g_loss` contains: `g_loss`.
    
    Example:

    .. code-block:: python3
    
        >>> trainer = dlt.train.BEGANTrainer(gen, disc, g_optim, d_optim, lambda_k, gamma)
        >>> # Training mode
        >>> trainer.train()
        >>> for batch, (prediction, loss) in trainer(train_data_loader):
        >>>     print(loss['d_loss']['measure'])
    """

    def __init__(self, generator, discriminator, g_optimizer, d_optimizer, lambda_k, gamma, d_iter=1):
        super(BEGANTrainer, self).__init__(generator, discriminator, g_optimizer, d_optimizer, d_iter)
        # Register losses
        self._losses['training'] = ['d_loss', 'real_loss', 'fake_loss', 'k', 'balance', 'measure', 'g_loss']
        self._losses['validation'] = ['g_loss']
        self.k = 0.0
        self.lambda_k = lambda_k
        self.gamma = gamma


    def d_step(self, g_input, real_input):
        disc, gen = self._models['discriminator'], self._models['generator']
        self._set_gradients('discriminator', True)
        self._set_gradients('generator', False)

        prediction = gen(g_input)

        fake_loss = (disc(prediction) - prediction).abs().mean()
        real_loss = (disc(real_input) - real_input).abs().mean()
        
        d_loss = real_loss - self.k*fake_loss
        
        disc.zero_grad()
        d_loss.backward()
        self._optimizers['discriminator'].step()

        balance = (self.gamma * real_loss.item() - fake_loss.item())
        self.k = min(max(self.k + self.lambda_k*balance, 0), 1)
        measure = real_loss.item()  + math.fabs(balance)

        ret_losses = {'d_loss': d_loss.item(),
                      'real_loss': real_loss.item(),
                      'fake_loss': fake_loss.item(),
                      'k': self.k, 'measure': measure, 'balance': balance}
        self.d_iter_counter += 1
        return prediction.data, ret_losses

    def g_step(self, g_input, real_input):
        disc, gen = self._models['discriminator'], self._models['generator']
        self._set_gradients('discriminator', False)
        self._set_gradients('generator', True)
            
        prediction = gen(g_input)
        loss = (disc(prediction) - prediction).abs().mean()
        
        if self.training:
            gen.zero_grad()
            loss.backward()
            self._optimizers['generator'].step()

        return prediction.data, {'g_loss': loss.item()}
