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
        add_loss (callable, optional): Extra loss term to be added to GAN
            objective (default None).

    Each iteration returns the mini-batch and a tuple containing:

        - The generator prediction.
        - A dictionary with:
            
            - `d_loss`, `real_loss`, `fake_loss` if training mode.
            - `g_loss` if validation mode.

    Example:
        >>> trainer = dlt.train.VanillaGANTrainer(gen, disc, g_optim, d_optim)
        >>> # Training mode
        >>> trainer.train()
        >>> for batch, (prediction, loss) in trainer(train_data_loader):
        >>>     print(loss['d_loss'])

    Warning:
        This trainer uses BCEWithLogitsLoss, which means that the discriminator
        must NOT have a sigmoid at the end.
    """
    def __init__(self, generator, discriminator, g_optimizer, d_optimizer, d_iter=1, add_loss=None):
        
        super(VanillaGANTrainer, self).__init__(generator, discriminator, g_optimizer, 
                                                d_optimizer, d_iter, add_loss)
        # Register losses
        self._losses['training'] = ['d_loss', 'fake_loss', 'real_loss']
        self._losses['validation'] = ['g_loss']
        self.real_label = 1
        self.fake_label = 0
        self.bce = torch.nn.BCEWithLogitsLoss()
        self.add_loss = add_loss

    def d_step(self, g_input, real_input):
        batch_size = g_input.size(0)
        for p in self.discriminator.parameters():
            p.requires_grad = True
        self.discriminator.zero_grad()
        prediction = Variable(self.generator(Variable(g_input, volatile=True)).data)
        real_label = Variable(prediction.data.new(batch_size,1).fill_(self.real_label))
        fake_label = Variable(prediction.data.new(batch_size,1).fill_(self.fake_label))

        loss_fake = self.bce(self.discriminator(prediction), fake_label)
        loss_real = self.bce(self.discriminator(Variable(real_input)), real_label)

        total_loss = loss_fake + loss_real
        if self.add_loss:
            total_loss = total_loss + self.add_loss(prediction, Variable(real_input))
        total_loss.backward()
        self.d_optimizer.step()

        ret_losses = {'d_loss': total_loss.data[0], 'real_loss':loss_real.data[0], 'fake_loss': loss_fake.data[0]}
        self.d_iter_counter += 1
        return prediction.data, ret_losses


    def g_step(self, g_input):
        batch_size = g_input.size(0)
        for p in self.discriminator.parameters():
            p.requires_grad = False
        if self.training:
            self.generator.zero_grad()
        prediction = self.generator(Variable(g_input, volatile=not self.training))
        # fake labels are real for generator cost
        d_prediction = self.discriminator(prediction)
        labelv = Variable(d_prediction.data.new(batch_size,1).fill_(self.real_label))
        error = self.bce(d_prediction, labelv)
        if self.training:
            error.backward()
            self.g_optimizer.step()
        return prediction.data, {'g_loss': error.data[0]}
