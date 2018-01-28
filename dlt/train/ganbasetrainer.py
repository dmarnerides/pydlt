from .basetrainer import BaseTrainer

class GANBaseTrainer(BaseTrainer):
    """Base Trainer to inherit functionality from for training Generative Adversarial Networks.
    
    Args:
        generator (nn.Module): The generator network.
        discriminator (nn.Module): The discriminator network.
        g_optimizer (torch.optim.Optimizer): Generator Optimizer.
        d_optimizer (torch.optim.Optimizer): Discriminator Optimizer.
        d_iter (int): Number of discriminator steps per generator step.
        add_loss (callable, optional): Extra loss term to be added to GAN
            objective.

    Inherits from :class:`dlt.train.BaseTrainer`
    """
    def __init__(self, generator, discriminator, g_optimizer, d_optimizer, d_iter, add_loss=None):
        super(GANBaseTrainer, self).__init__()
        # Register models
        self._models['generator'] = generator
        self._models['discriminator'] = discriminator
        self.generator = generator
        self.discriminator = discriminator
        self.g_optimizer = g_optimizer
        self.d_optimizer = d_optimizer
        self.d_iter = d_iter
        self.d_iter_counter = 0
        self.add_loss = add_loss

    def iteration(self, data):
        if self.training:
            ret = self.d_step(data[0], data[1])
            if self.d_iter_counter % self.d_iter == 0:
                self.g_step(data[0])
            return ret
        else:
            return self.g_step(data[0])

    def d_step(self, g_input, real_input):
        raise NotImplementedError

    def g_step(self, g_input):
        raise NotImplementedError
