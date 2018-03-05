import torch
from torch import autograd
from torch.autograd import Variable
from .ganbasetrainer import GANBaseTrainer
from ..util.misc import _get_scalar_value

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
        add_loss (callable, optional): Extra loss term to be added to GAN
            objective (default None).

    Each iteration returns the mini-batch and a tuple containing:

        - The generator prediction.
        - A dictionary with:
            
            - `d_loss`, `w_loss`, `gp` if training mode.
            - `g_loss` if validation mode.

    Example:
        >>> trainer = dlt.train.WGANGPTrainer(gen, disc, g_optim, d_optim, lambda_gp)
        >>> # Training mode
        >>> trainer.train()
        >>> for batch, (prediction, loss) in trainer(train_data_loader):
        >>>     print(loss['w_loss'])
    """
    def __init__(self, generator, discriminator, g_optimizer, d_optimizer, lambda_gp, d_iter=1, add_loss=None):
        super(WGANGPTrainer, self).__init__(generator, discriminator, g_optimizer, 
                                                d_optimizer, d_iter, add_loss)
        # Register losses
        self._losses['training'] = ['w_loss', 'd_loss', 'gp']
        self._losses['validation'] = ['g_loss']
        self.lambda_gp = lambda_gp
        self.alpha = None
        self.gradout = None
                
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
        error_fake = self.discriminator(prediction).mean()
        error_real = self.discriminator(Variable(real_input)).mean()
        gp = self.get_gp(prediction.data, real_input)

        w_loss = error_fake - error_real
        total_loss = w_loss + gp

        if self.add_loss:
            total_loss = total_loss + self.add_loss(prediction, Variable(real_input))
        
        total_loss.backward()
        self.d_optimizer.step()

        ret_losses = {'w_loss': _get_scalar_value(w_loss.data),
                      'gp': _get_scalar_value(gp.data),
                      'd_loss': _get_scalar_value(total_loss.data)}
        self.d_iter_counter += 1
        return prediction.data, ret_losses

    def g_step(self, g_input):
        for p in self.discriminator.parameters():
            p.requires_grad = False
        if self.training:
            self.generator.zero_grad()
            prediction = self.generator(Variable(g_input))
            error = - self.discriminator(prediction).mean()
            error.backward()
            self.g_optimizer.step()
        else:
            if self._use_no_grad:
                with torch.no_grad():
                    prediction = self.generator(Variable(g_input))
                    error = - self.discriminator(prediction).mean()
            else:
                prediction = self.generator(Variable(g_input, volatile=True))
                error = - self.discriminator(prediction).mean()
        return prediction.data, {'g_loss': _get_scalar_value(error.data)}


    def make_alpha(self, real_input):
        dimensions = [real_input.size(0), *[1 for x in range(real_input.ndimension() - 1)]]
        if self.alpha is None:
            self.alpha = real_input.new(*dimensions).uniform_()
        else:
            self.alpha.resize_(*dimensions).uniform_()

    def make_grad_out(self, t_disc_interpolates):
        if self.gradout is None:
            self.gradout = t_disc_interpolates.clone().fill_(1)
        else:
            self.gradout.resize_(t_disc_interpolates.size()).fill_(1)

    def get_gp(self, fake_input, real_input):
        self.make_alpha(real_input)
        interpolates = Variable(self.alpha * real_input + ((1 - self.alpha) * fake_input), requires_grad=True)
        disc_interpolates = self.discriminator(interpolates)
        self.make_grad_out(disc_interpolates.data)
        gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                                  grad_outputs=self.gradout,
                                  create_graph=True, retain_graph=True, only_inputs=True)[0]
        gradients = gradients.view(gradients.size(0), -1)
        gradient_penalty = torch.mean((1. - torch.sqrt(1e-8+torch.sum(gradients**2, dim=1)))**2)*self.lambda_gp
        return gradient_penalty