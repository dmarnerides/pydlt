from torch import nn

def selu_init(model):
    for m in model.modules():
        if any([isinstance(m, x) for x in [nn.Conv2d, nn.ConvTranspose2d, nn.Linear]]):
            nn.init.kaiming_normal_(m.weight, 1)
            if m.bias is not None:
                nn.init.constant_(m.bias, val=0)

class Generator(nn.Module):
    def __init__(self, num_hidden, z_dim, num_chan, num_pix):
        super(Generator, self).__init__()
        self.num_pix = num_pix
        self.num_chan = num_chan
        self.main = nn.Sequential(
            nn.Linear(z_dim, num_hidden),
            nn.SELU(),
            nn.Linear(num_hidden, num_hidden),
            nn.SELU(),
            nn.Linear(num_hidden, num_chan*num_pix*num_pix),
            nn.Tanh()
        )
        selu_init(self)

    def forward(self, v_input):
        return self.main(v_input).view(v_input.size(0), self.num_chan, self.num_pix, self.num_pix)

class Discriminator(nn.Module):
    def __init__(self, num_hidden, num_chan, num_pix):
        super(Discriminator, self).__init__()
        self.num_pix = num_pix
        self.num_chan = num_chan
        self.main = nn.Sequential(
            nn.Linear(num_chan*num_pix*num_pix, num_hidden),
            nn.SELU(),
            nn.Linear(num_hidden, num_hidden),
            nn.SELU()
        )
        self.last_layer = nn.Linear(num_hidden, 1)
        selu_init(self)

    # The correction term is for WGAN-CT
    def forward(self, v_input, correction_term=False):
        if correction_term:
            main = self.main(v_input.view(v_input.size(0), -1))
            noisy_main = nn.functional.dropout(main, p=0.1)
            return main, self.last_layer(noisy_main)
        else:
            return self.last_layer(self.main(v_input.view(v_input.size(0), -1)))

# BEGAN needs an autoencoding discriminator
class DiscriminatorBEGAN(nn.Module):
    def __init__(self, num_hidden, num_chan, num_pix):
        super(DiscriminatorBEGAN, self).__init__()
        self.num_pix = num_pix
        self.num_chan = num_chan
        self.main = nn.Sequential(
            nn.Linear(num_chan*num_pix*num_pix, num_hidden), nn.SELU(),
            nn.Linear(num_hidden, num_hidden), nn.SELU(),
            nn.Linear(num_hidden, num_chan*num_pix*num_pix),
        )
        selu_init(self)

    def forward(self, v_input):
        res = self.main(v_input.view(v_input.size(0), -1))
        return res.view(v_input.size(0), self.num_chan, self.num_pix, self.num_pix)