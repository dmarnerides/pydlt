import torch
from torch import autograd
from torch.autograd import Variable
from .ganbasetrainer import GANBaseTrainer

class WGANCTTrainer(GANBaseTrainer):
    """Wasserstein GAN Trainer with gradient penalty and correction term.
       
       From Improving the Improved Training of Wasserstein GANs: A Consistency
       Term and Its Dual Effect.

       https://openreview.net/forum?id=SJx9GQb0-
    
    Args:
        generator (nn.Module): The generator network.
        discriminator (nn.Module): The discriminator network.
        g_optimizer (torch.optim.Optimizer): Generator Optimizer.
        d_optimizer (torch.optim.Optimizer): Discriminator Optimizer.
        lambda_gp (float): Weight of gradient penalty.
        m_ct (float): Constant bound for consistency term.
        lambda_ct (float): Weight of consistency term.
        d_iter (int, optional): Number of discriminator steps per generator
            step (default 1).

    Each iteration returns the mini-batch and a tuple containing:

        - The generator prediction.
        - A dictionary containing a `d_loss` (not when validating) and a 
          `g_loss` dictionary (only if a generator step is performed):
            
            - `d_loss contains`: `d_loss`, `w_loss`, `gp` and `ct`.
            - `g_loss` contains: `g_loss`.

    Warning:

        The discriminator forward function needs to be able to accept an optional
        bool argument `correction_term`. When set to true, the forward function
        must add dropout noise to the model and return a tuple containing the 
        second to last output of the discriminator along with the final output.


    Example:
        >>> trainer = dlt.train.WGANCTTrainer(gen, disc, g_optim, d_optim, lambda_gp, m_ct, lambda_ct)
        >>> # Training mode
        >>> trainer.train()
        >>> for batch, (prediction, loss) in trainer(train_data_loader):
        >>>     print(loss['d_loss']['w_loss'])
    """
    def __init__(self, generator, discriminator, g_optimizer, d_optimizer, lambda_gp, m_ct, lambda_ct, d_iter=1):
        super(WGANCTTrainer, self).__init__(generator, discriminator, g_optimizer, 
                                                d_optimizer, d_iter)
        # Register losses
        self._losses['training'] = ['w_loss', 'd_loss', 'gp', 'ct', 'g_loss']
        self._losses['validation'] = ['g_loss']
        self.lambda_gp = lambda_gp
        self.m_ct = m_ct
        self.lambda_ct = lambda_ct
        
    def d_step(self, g_input, real_input):
        disc, gen = self._models['discriminator'], self._models['generator']
        self._set_gradients('discriminator', True)
        self._set_gradients('generator', False)

        
        prediction = gen(g_input)
        error_fake = disc(prediction).mean()
        error_real = disc(Variable(real_input)).mean()

        gp = self.get_gp(prediction, real_input)
        ct = self.get_ct(real_input)
        w_loss = error_fake - error_real
        total_loss = w_loss + gp + ct

        disc.zero_grad()
        total_loss.backward()
        self._optimizers['discriminator'].step()

        ret_losses = {'w_loss': w_loss.item(), 'gp': gp.item(),
                      'ct': ct.item(), 'd_loss': total_loss.item()}
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

    def l2_norm(self, x):
        return x.pow(2).view(x.size(0), -1).sum(-1).add(1e-8).sqrt()

    def get_ct(self, real_input):
        dx_dash_n2last, dx_dash = self._models['discriminator'](real_input, correction_term=True)
        dx_dashdash_n2last, dx_dashdash = self._models['discriminator'](real_input, correction_term=True)
        res = self.l2_norm(dx_dash - dx_dashdash) + 0.1 \
              * self.l2_norm(dx_dash_n2last - dx_dashdash_n2last) \
              - self.m_ct
        return torch.nn.functional.relu(res, 0).mean()*self.lambda_ct
