import torch
import dlt
from dlt.train import VanillaGANTrainer as GAN
from dlt.train import WGANGPTrainer as WGANGP
from dlt.train import WGANCTTrainer as WGANCT
from dlt.train import BEGANTrainer as BEGAN
from dlt.train import FisherGANTrainer as FisherGAN
from models import *

# Settings
dlt.config.make_subsets({'model': ['generator', 'discriminator'],
                         'optimizer': ['generator', 'discriminator']})
dlt.config.add_extras([
    dict(flag='--gan_type', choices=['vanilla', 'wgan-gp', 'wgan-ct', 'began', 'fishergan'],
                            default='vanilla', help='Gan type'),
    dict(flag='--num_hidden', type=int, default=64, help='Number of hidden units'),
    dict(flag='--z_dim', type=int, default=128, help='Input noise dimensionality'),
    dict(flag='--lambda_gp', type=float, default=10, help='Gradient penalty magnitude'),
    dict(flag='--m_ct', type=float, default=0.001, help='Constant bound for consistency term for WGAN-CT'),
    dict(flag='--lambda_ct', type=float, default=0.001, help='Weight of consistency term for WGAN-CT'),
    dict(flag='--lambda_k', type=float, default=0.001, help='Learning rate for k for BEGAN'),
    dict(flag='--gamma', type=float, default=0.5, help='Gamma for BEGAN (diversity ratio)'),
    dict(flag='--rho', type=float, default=1e-6, help='rho for FisherGAN'),
    dict(flag='--d_iter', type=int, default=2, help='Number of discriminator steps per generator')
])
opt = dlt.config.parse(verbose=True)

# Configure seeds
if opt.seed is not None:
    torch.manual_seed(opt.seed)

# Data
sizes = {'mnist': (1, 28), 'fashionmnist': (1, 28),
         'cifar10': (3, 32), 'cifar100': (3, 32)}
if opt.torchvision_dataset not in sizes:
    raise ValueError('--torchvision_dataset must be one of {0}'.format(','.join(sizes.keys())))
size = sizes[opt.torchvision_dataset]
def preprocess(datum):
    noise = torch.Tensor(opt.z_dim).uniform_(-1, 1)
    real_image = (dlt.util.cv2torch(datum[0]).float()/255.0) * 1.8 - 0.9
    # By convention, the trainer accepts the first point as the generator
    # input and the second as the real input for the discriminator
    return noise, real_image

dataset = dlt.config.torchvision_dataset()
loader = dlt.config.loader(dataset, preprocess)

# Models
generator = Generator(opt.num_hidden, opt.z_dim, size[0], size[1])
gen_chkp = dlt.config.model_checkpointer(generator, subset='generator')

if opt.gan_type == 'began':
    discriminator = DiscriminatorBEGAN(opt.num_hidden, size[0], size[1])    
else:
    discriminator = Discriminator(opt.num_hidden, size[0], size[1])
disc_chkp = dlt.config.model_checkpointer(discriminator, subset='discriminator')

# Cudafy
if opt.use_gpu:
    torch.cuda.set_device(opt.device)
    torch.backends.cudnn.benchmark = opt.cudnn_benchmark
    generator.cuda()
    discriminator.cuda()

# Optimizers
g_optim = dlt.config.optimizer(generator, subset='generator')
g_optim_chkp = dlt.config.optimizer_checkpointer(g_optim, subset='generator')
d_optim = dlt.config.optimizer(discriminator, subset='discriminator')
d_optim_chkp = dlt.config.optimizer_checkpointer(d_optim, subset='discriminator')

# Trainer
if opt.gan_type == 'wgan-gp':
    trainer = WGANGP(generator, discriminator, g_optim, d_optim, opt.lambda_gp, opt.d_iter)
elif opt.gan_type == 'began':
    trainer = BEGAN(generator, discriminator, g_optim, d_optim, opt.lambda_k, opt.gamma, opt.d_iter)
elif opt.gan_type == 'fishergan':
    trainer = FisherGAN(generator, discriminator, g_optim, d_optim, opt.rho, opt.d_iter)
elif opt.gan_type == 'wgan-ct':
    trainer = WGANCT(generator, discriminator, g_optim, d_optim, opt.lambda_gp, opt.m_ct, opt.lambda_ct, opt.d_iter)
else:
    trainer = GAN(generator, discriminator, g_optim, d_optim, opt.d_iter)

trainer_chkp = dlt.config.trainer_checkpointer(trainer)

if opt.use_gpu:
    trainer.cuda() # Trainers might have buffers that need to be transferred to GPU

# Logging
log = dlt.util.Logger('training', trainer.loss_names_training(), opt.save_path)

# Training loop
for epoch in range(trainer.epoch, opt.max_epochs):
    tag = 'epoch-{0}'.format(epoch)
    print('-'*79 + '\nEpoch {0}:'.format(epoch))
    # Set to training mode
    trainer.train()
    # The trainer iterator performs the optimization and gives predictions and
    # losses at each iteration
    for i, (batch, (prediction, losses)) in enumerate(trainer(loader)):
        # Show progress of each iteration and log the losses
        dlt.config.sample_images([batch[1], prediction], color=size[0] == 3,
                                 preprocess=dlt.util.map_range, tag=tag)
        log(losses)
    
    # Checkpoint everything
    gen_chkp(generator, tag=tag)
    disc_chkp(discriminator, tag=tag)
    g_optim_chkp(g_optim, tag=tag)
    d_optim_chkp(d_optim, tag=tag)
    trainer_chkp(trainer, tag=tag)
    