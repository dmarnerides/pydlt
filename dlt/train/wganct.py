import torch
from torch import autograd
from torch.autograd import Variable
from .ganbasetrainer import GANBaseTrainer
from ..util.misc import _get_scalar_value

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
        add_loss (callable, optional): Extra loss term to be added to GAN
            objective (default None).

    Each iteration returns the mini-batch and a tuple containing:

        - The generator prediction.
        - A dictionary containing a `d_loss` (not when validating) and a 
          `g_loss` dictionary (only if a generator step is performed):
            
            - `d_loss contains`: `d_loss`, `w_loss`, `gp` and `ct`.
            - `g_loss` contains: `g_loss` (and extra_loss if add_loss is used).

    Warning:

        The discriminator needs to have a member function `get_get_ct_results`
        which returns the second to last output of the discriminator along with
        the last element.


    Example:
        >>> trainer = dlt.train.WGANGPTrainer(gen, disc, g_optim, d_optim, lambda_gp)
        >>> # Training mode
        >>> trainer.train()
        >>> for batch, (prediction, loss) in trainer(train_data_loader):
        >>>     print(loss['d_loss']['w_loss'])
    """
    def __init__(self, generator, discriminator, g_optimizer, d_optimizer, lambda_gp, m_ct, lambda_ct, d_iter=1, add_loss=None):
        super(WGANCTTrainer, self).__init__(generator, discriminator, g_optimizer, 
                                                d_optimizer, d_iter, add_loss)
        # Register losses
        self._losses['training'] = ['w_loss', 'd_loss', 'gp', 'ct']
        self._losses['validation'] = ['g_loss']
        self.lambda_gp = lambda_gp
        self.m_ct = m_ct
        self.lambda_ct = lambda_ct
        self.alpha = None
        self.gradout = None
        if self.add_loss is not None:
            self._losses['training'] += ['extra_loss']
        
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
        ct = self.get_ct(real_input)
        w_loss = error_fake - error_real
        total_loss = w_loss + gp + ct

        total_loss.backward()
        self.d_optimizer.step()

        ret_losses = {'w_loss': _get_scalar_value(w_loss.data),
                      'gp': _get_scalar_value(gp.data),
                      'ct': _get_scalar_value(ct.data),
                      'd_loss': _get_scalar_value(total_loss.data)}
        self.d_iter_counter += 1
        return prediction.data, ret_losses

    def g_step(self, g_input, real_input):
        for p in self.discriminator.parameters():
            p.requires_grad = False
        if self.training:
            self.generator.zero_grad()
            prediction = self.generator(Variable(g_input))
            error = - self.discriminator(prediction).mean()
            total_loss = error
            if self.add_loss:
                extra_loss = self.add_loss(prediction, Variable(real_input))
                total_loss += extra_loss
            total_loss.backward()
            self.g_optimizer.step()
        else:
            if self._use_no_grad:
                with torch.no_grad():
                    prediction = self.generator(Variable(g_input))
                    error = - self.discriminator(prediction).mean()
                    total_loss = error
                    if self.add_loss:
                        extra_loss = self.add_loss(prediction, Variable(real_input))
                        total_loss += extra_loss
            else:
                prediction = self.generator(Variable(g_input, volatile=True))
                error = - self.discriminator(prediction).mean()
                total_loss = error
                if self.add_loss:
                    extra_loss = self.add_loss(prediction, Variable(real_input))
                    total_loss += extra_loss
        ret_loss = {'g_loss': _get_scalar_value(error.data)}
        if self.add_loss:
            ret_loss['extra_loss'] = _get_scalar_value(extra_loss.data)
        return prediction.data, ret_loss


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

    def l2_norm(self, x, y):
        return ((x - y)**2).view(x.size(0), -1).sum(-1).add(1e-8).sqrt()


    def get_ct(self, real_input):
        dx_dash_n2last, dx_dash = self.discriminator.get_ct_results(Variable(real_input))
        dx_dashdash_n2last, dx_dashdash = self.discriminator.get_ct_results(Variable(real_input))
        res = self.l2_norm(dx_dash, dx_dashdash) + 0.1 \
              * self.l2_norm(dx_dash_n2last, dx_dashdash_n2last) \
              - self.m_ct
        return torch.nn.functional.relu(res, 0).mean()*self.lambda_ct
