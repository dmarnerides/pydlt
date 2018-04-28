import torch
from torch import autograd
from torch.autograd import Variable
from .ganbasetrainer import GANBaseTrainer

class WGANGPTrainer(GANBaseTrainer):
    """Wasserstein GAN Trainer with gradient penalty. 
    
    Args:
        generator (nn.Module): The generator network.
        discriminator (nn.Module): The discriminator network.
        g_optimizer (torch.optim.Optimizer): Generator Optimizer.
        d_optimizer (torch.optim.Optimizer): Discriminator Optimizer.
        lambda_gp (float): Weight of gradient penalty.
        d_iter (int, optional): Number of discriminator steps per generator
            step (default 1).

    Each iteration returns the mini-batch and a tuple containing:

        - The generator prediction.
        - A dictionary containing a `d_loss` (not when validating) and a 
          `g_loss` dictionary (only if a generator step is performed):
            
            - `d_loss contains`: `d_loss`, `w_loss`, and `gp`.
            - `g_loss` contains: `g_loss`.

    Example:
        >>> trainer = dlt.train.WGANGPTrainer(gen, disc, g_optim, d_optim, lambda_gp)
        >>> # Training mode
        >>> trainer.train()
        >>> for batch, (prediction, loss) in trainer(train_data_loader):
        >>>     print(loss['d_loss']['w_loss'])
    """
    def __init__(self, generator, discriminator, g_optimizer, d_optimizer, lambda_gp, d_iter=1):
        super(WGANGPTrainer, self).__init__(generator, discriminator, g_optimizer, 
                                                d_optimizer, d_iter)
        # Register losses
        self._losses['training'] = ['w_loss', 'd_loss', 'gp', 'g_loss']
        self._losses['validation'] = ['g_loss']
        self.lambda_gp = lambda_gp
                
    def d_step(self, g_input, real_input):
        disc, gen = self._models['discriminator'], self._models['generator']
        self._set_gradients('discriminator', True)
        self._set_gradients('generator', False)

        
        prediction = gen(g_input)
        error_fake = disc(prediction).mean()
        error_real = disc(Variable(real_input)).mean()

        gp = self.get_gp(prediction, real_input)
        w_loss = error_fake - error_real
        total_loss = w_loss + gp

        disc.zero_grad()
        total_loss.backward()
        self._optimizers['discriminator'].step()

        ret_losses = {'w_loss': w_loss.item(), 'gp': gp.item(), 'd_loss': total_loss.item()}
        self.d_iter_counter += 1
        return prediction, ret_losses

    def g_step(self, g_input, real_input):
        disc, gen = self._models['discriminator'], self._models['generator']
        self._set_gradients('discriminator', False)
        self._set_gradients('generator', True)

        prediction = gen(Variable(g_input))
        loss = - disc(prediction).mean()
        
        if self.training:    
            gen.zero_grad()
            loss.backward()
            self._optimizers['generator'].step()

        return prediction, {'g_loss': loss.item()}

    def get_gp(self, fake_input, real_input):
        dimensions = [real_input.size(0)] + [1] * (real_input.ndimension() - 1)
        alpha = torch.Tensor(*dimensions).to(real_input.device).uniform_()
        interpolates = alpha * real_input + ((1 - alpha) * fake_input)
        interpolates.requires_grad_(True)
        disc_interpolates = self._models['discriminator'](interpolates)

        gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                                  grad_outputs=torch.ones_like(disc_interpolates),
                                  create_graph=True, retain_graph=True, only_inputs=True)[0]
        gradients = gradients.view(gradients.size(0), -1)
        gradient_penalty = torch.mean((1. - torch.sqrt(1e-8+torch.sum(gradients**2, dim=1)))**2)*self.lambda_gp
        return gradient_penalty