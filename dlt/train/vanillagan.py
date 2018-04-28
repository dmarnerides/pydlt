import torch
from torch.autograd import Variable
from .ganbasetrainer import GANBaseTrainer

class VanillaGANTrainer(GANBaseTrainer):
    """Generative Adversarial Networks trainer. 
        
    Args:
        generator (nn.Module): The generator network.
        discriminator (nn.Module): The discriminator network.
        g_optimizer (torch.optim.Optimizer): Generator Optimizer.
        d_optimizer (torch.optim.Optimizer): Discriminator Optimizer.
        d_iter (int, optional): Number of discriminator steps per generator
            step (default 1).

    Each iteration returns the mini-batch and a tuple containing:

        - The generator prediction.
        - A dictionary containing a `d_loss` (not when validating) and a 
          `g_loss` dictionary (only if a generator step is performed):
            
            - `d_loss contains`: `d_loss`, `real_loss`, and `fake_loss`.
            - `g_loss` contains: `g_loss`.

    Example:
        >>> trainer = dlt.train.VanillaGANTrainer(gen, disc, g_optim, d_optim)
        >>> # Training mode
        >>> trainer.train()
        >>> for batch, (prediction, loss) in trainer(train_data_loader):
        >>>     print(loss['d_loss']['d_loss'])

    Warning:
        This trainer uses BCEWithLogitsLoss, which means that the discriminator
        must NOT have a sigmoid at the end.
    """
    def __init__(self, generator, discriminator, g_optimizer, d_optimizer, d_iter=1):
        super(VanillaGANTrainer, self).__init__(generator, discriminator, g_optimizer, 
                                                d_optimizer, d_iter)
        # Register losses
        self._losses['training'] = ['d_loss', 'fake_loss', 'real_loss', 'g_loss']
        self._losses['validation'] = ['g_loss']
        self.bce = torch.nn.BCEWithLogitsLoss()

    def d_step(self, g_input, real_input):
        batch_size = g_input.size(0)
        disc, gen = self._models['discriminator'], self._models['generator']
        self._set_gradients('discriminator', True)
        self._set_gradients('generator', False)

        prediction = gen(g_input)

        real_label = torch.ones(batch_size, 1, device=prediction.device)
        fake_label = torch.zeros(batch_size, 1, device=prediction.device)

        loss_fake = self.bce(disc(prediction), fake_label)
        loss_real = self.bce(disc(real_input), real_label)

        total_loss = loss_fake + loss_real
        
        disc.zero_grad()
        total_loss.backward()
        self._optimizers['discriminator'].step()

        ret_losses = {'d_loss': total_loss.item(), 
                      'real_loss': loss_real.item(), 
                      'fake_loss': loss_fake.item()}
        self.d_iter_counter += 1
        return prediction, ret_losses


    def g_step(self, g_input, real_input):
        batch_size = g_input.size(0)
        disc, gen = self._models['discriminator'], self._models['generator']
        self._set_gradients('discriminator', False)
        self._set_gradients('generator', True)

        prediction = gen(g_input)
        # fake labels are real for generator cost (optimization trick from GAN paper)
        d_prediction = disc(prediction)
        real_label = torch.ones(batch_size, 1, device=d_prediction.device)
        loss = self.bce(d_prediction, real_label)
        
        if self.training:
            gen.zero_grad()
            loss.backward()
            self._optimizers['generator'].step()

        return prediction, {'g_loss': loss.item()}
