from .basetrainer import BaseTrainer

class GANBaseTrainer(BaseTrainer):
    """Base Trainer to inherit functionality from for training Generative Adversarial Networks.
    
    Args:
        generator (nn.Module): The generator network.
        discriminator (nn.Module): The discriminator network.
        g_optimizer (torch.optim.Optimizer): Generator Optimizer.
        d_optimizer (torch.optim.Optimizer): Discriminator Optimizer.
        d_iter (int): Number of discriminator steps per generator step.

    Inherits from :class:`dlt.train.BaseTrainer`
    """
    def __init__(self, generator, discriminator, g_optimizer, d_optimizer, d_iter):
        super(GANBaseTrainer, self).__init__()
        # Register models
        self._models['generator'] = generator
        self._models['discriminator'] = discriminator
        self._optimizers['generator'] = g_optimizer
        self._optimizers['discriminator'] = d_optimizer

        self.d_iter = d_iter
        self.d_iter_counter = 0

    def iteration(self, data):
        if self.training:
            pred, losses = self.d_step(data[0], data[1])
            if self.d_iter_counter % self.d_iter == 0:
                _, g_loss = self.g_step(data[0], data[1])
                losses.update(g_loss)
        else:
            pred, losses = self.g_step(data[0], data[1])
        return pred, losses

    def d_step(self, g_input, real_input):
        raise NotImplementedError

    def g_step(self, g_input, real_input):
        raise NotImplementedError

    def _set_gradients(self, model, value):
        for p in self._models[model].parameters():
            p.requires_grad_(value)