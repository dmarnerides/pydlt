import torch
import dlt
from dlt.train import VanillaGANTrainer as GAN
from dlt.train import WGANGPTrainer as WGANGP
from dlt.train import BEGANTrainer as BEGAN
from dlt.train import FisherGANTrainer as FisherGAN
from models import *

# Settings
dlt.config.make_subsets({'model': ['generator', 'discriminator'],
                         'optimizer': ['generator', 'discriminator']})
dlt.config.add_extras([
    dict(flag='--gan_type', choices=['vanilla', 'wgan-gp', 'began', 'fishergan'], default='vanilla', help='Gan type'),
    dict(flag='--num_hidden', type=int, default=64, help='Number of hidden units'),
    dict(flag='--z_dim', type=int, default=128, help='Input noise dimensionality'),
    dict(flag='--lambda_gp', type=float, default=10, help='Gradient penalty magnitude'),
    dict(flag='--lambda_k', type=float, default=0.001, help='Learning rate for k for BEGAN'),
    dict(flag='--gamma', type=float, default=0.5, help='Gamma for BEGAN (diversity ratio)'),
    dict(flag='--rho', type=float, default=1e-6, help='rho for FisherGAN'),
    dict(flag='--d_iter', type=int, default=2, help='Number of discriminator steps per generator'),
    dict(flag='--show_progress', type=dlt.util.str2bool, default=True, help='Show samples while training'),
])
opt = dlt.config.parse()

# Configure seeds
if opt.seed is not None:
    torch.manual_seed(opt.seed)

# Data
sizes = {'mnist': (1,28), 'fashionmnist': (1,28),
         'cifar10': (3,32), 'cifar100': (3,32)}
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
d_optim = dlt.config.optimizer(discriminator, subset='discriminator')

# Trainer
if opt.gan_type == 'wgan-gp':
    trainer = WGANGP(generator, discriminator, g_optim, d_optim, opt.lambda_gp, opt.d_iter)
elif opt.gan_type == 'began':
    trainer = BEGAN(generator, discriminator, g_optim, d_optim, opt.lambda_k, opt.gamma, opt.d_iter)
elif opt.gan_type == 'fishergan':
    trainer = FisherGAN(generator, discriminator, g_optim, d_optim, opt.rho, opt.d_iter)
else:
    trainer = GAN(generator, discriminator, g_optim, d_optim, opt.d_iter)

if opt.use_gpu:
    trainer.cuda() # Trainers might have buffers that need to be transferred to GPU

# Logging
log = dlt.util.Logger('training', trainer.loss_names_training(), opt.save_path)
# epoch checkpoint
epoch_chkp, current_epoch = dlt.config.epoch_checkpointer()


# Training loop
for epoch in range(current_epoch, opt.max_epochs):
    print('-'*79 + '\nEpoch {0}:'.format(epoch))
    trainer.train()
    for i, (batch, (prediction, losses)) in enumerate(trainer(loader)):
        # Show progress
        if opt.show_progress and i % 100 == 0:
            imgs = [dlt.util.map_range(x) for x in [batch[1], prediction]]
            img_grid = dlt.util.make_grid(imgs, color=size[0] == 3)
            dlt.viz.imshow(img_grid, pause=0.02, title='Epoch {0}, Iteration {1}'.format(epoch, i+1))
        # Log the losses
        log(losses['d_loss'])
    # Do some checkpointomg
    epoch_chkp(epoch + 1) # +1 because this epoch has finished
    gen_chkp(generator, tag='epoch-{0}'.format(epoch))
    disc_chkp(discriminator, tag='epoch-{0}'.format(epoch))