import math
from torch import nn

def selu_init(model):
    for m in model.modules():
        if hasattr(m,'pretrained') and m.pretrained:
            continue
        if any([isinstance(m, x) for x in [nn.Conv2d, nn.ConvTranspose2d, nn.Linear]]):
            fan_in = nn.init._calculate_correct_fan(m.weight.data, 'fan_in')
            nn.init.normal(m.weight.data, std=1.0/math.sqrt(fan_in))

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
            nn.SELU(),
            nn.Linear(num_hidden, 1),
        )
        selu_init(self)

    def forward(self, v_input):
        return self.main(v_input.view(v_input.size(0), -1))

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