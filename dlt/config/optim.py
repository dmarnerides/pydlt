from itertools import chain
import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR, LambdaLR
from ..util import Checkpointer
from .opts import fetch_opts, parse

def optimizer(model, extra_params=None, subset=None):
    """Returns the optimizer for the given model.
    
    Args:
        model (nn.Module): The network for the optimizer.
        extra_params (generator, optional): Extra parameters to pass to the optimizer.
        subset (string, optional): Specifies the subset of the relevant
            categories, if any of them was split (default, None).

    Relevant Command Line Arguments:

        - **optimizer**: `--optimizer`, `--lr`, `--momentum`,
            `--dampening`, `--beta1`, `--beta2`, `--weight_decay`.

    Note:
        Settings are automatically acquired from a call to :func:`dlt.config.parse`
        from the built-in ones. If :func:`dlt.config.parse` was not called in the 
        main script, this function will call it.
    """
    opts = fetch_opts(categories=['general', 'optimizer'], subset=subset)

    if opts.optimizer not in parse.optimizers:
        raise ValueError('Optimizer {0} not available.'.format(opts.optimizer))

    grad_params = filter(lambda p: p.requires_grad, model.parameters())
    if extra_params is not None:
        grad_params = chain(grad_params, filter(lambda p: p.requires_grad, extra_params))

    if opts.optimizer == 'adam':
        ret_optimizer = torch.optim.Adam(grad_params, lr=opts.lr, 
            betas=(opts.beta1, opts.beta2), weight_decay=opts.weight_decay)
    elif opts.optimizer == 'sgd':
        ret_optimizer = torch.optim.SGD(grad_params, lr=opts.lr, momentum=opts.momentum, 
            dampening=opts.dampening, weight_decay=opts.weight_decay)
    elif opts.optimizer == 'adadelta':
        ret_optimizer = torch.optim.Adadelta(grad_params, lr=opts.lr, rho=opts.rho, 
            eps=opts.optim_eps, weight_decay=opts.weight_decay)
    elif opts.optimizer == 'adagrad':
        ret_optimizer = torch.optim.Adagrad(grad_params, lr=opts.lr, 
            lr_decay=opts.lr_decay, weight_decay=opts.weight_decay)
    elif opts.optimizer == 'sparseadam':
        ret_optimizer = torch.optim.SparseAdam(grad_params, lr=opts.lr, 
            betas=(opts.beta1, opts.beta2), eps=opts.optim_eps)
    elif opts.optimizer == 'adamax':
        ret_optimizer = torch.optim.Adamax(grad_params, lr=opts.lr, 
            betas=(opts.beta1, opts.beta2), eps=opts.optim_eps, weight_decay=opts.weight_decay)
    elif opts.optimizer == 'rmsprop':
        ret_optimizer = torch.optim.RMSprop(grad_params, lr=opts.lr,
            alpha=opts.alpha, eps=opts.optim_eps, weight_decay=opts.weight_decay,
            momentum=opts.momentum, centered=opts.centered)

    return ret_optimizer

def optimizer_checkpointer(optimizer, subset=None):
    """Returns the optimizer checkpointer. Configurable using command line arguments.
    
    The function also loads any previous checkpoints if present.

    Args:
        optimizer (torch.optim.Optimizer): The optimizer.
        subset (string, optional): Specifies the subset of the relevant
            categories, if any of them was split (default, None).

    Relevant Command Line Arguments:

        - **general**: `--experiment_name`, `--save_path`.
        - **model**: `--overwrite_optimizer_chkp`, `--timestamp_optimizer_chkp`,
            `--count_optimizer_chkp`.

    Note:
        Settings are automatically acquired from a call to :func:`dlt.config.parse`
        from the built-in ones. If :func:`dlt.config.parse` was not called in the 
        main script, this function will call it.
    """

    opts = fetch_opts(['general', 'optimizer'], subset)
    name = subset['optimizer'] if isinstance(subset, dict) else subset
    exp_name = opts.experiment_name + '_' if opts.experiment_name != '' else ''
    optim_chkp = Checkpointer('{0}{1}optimizer'.format(exp_name, '' if name is None else name + '_'),
                                  directory=opts.save_path, overwrite=opts.overwrite_optimizer_chkp,
                                  timestamp=opts.timestamp_optimizer_chkp, add_count=opts.count_optimizer_chkp)
    optim_chkp.load(optimizer)
    return optim_chkp


def scheduler(optimizer, subset=None):
    """Returns a scheduler callable closure which accepts one argument.
    
    Configurable using command line arguments.
    
    Args:
        optimizer (torch.optim.Optimizer): The optimizer for the scheduler.
        subset (string, optional): Specifies the subset of the relevant
            categories, if any of them was split (default, None).

    Relevant Command Line Arguments:

        - **scheduler**: `--lr_schedule`, `--lr_step_size`, `--lr_patience`,
            `--lr_cooldown`, `--lr_ratio`, `--lr_min`,

    Note:
        Settings are automatically acquired from a call to :func:`dlt.config.parse`
        from the built-in ones. If :func:`dlt.config.parse` was not called in the 
        main script, this function will call it.
    """
    opts = fetch_opts(categories=['scheduler'], subset=subset)
    if opts.lr_schedule == 'plateau':
        ret_scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=opts.lr_ratio, threshold=0.0001,
                                    patience=opts.lr_patience, verbose=True, threshold_mode='rel',
                                    cooldown=opts.lr_cooldown, min_lr=opts.lr_min, eps=1e-08)
    elif opts.lr_schedule == 'step':
        ret_scheduler = StepLR(optimizer, step_size=opts.lr_step_size, gamma=opts.lr_ratio)
    elif opts.lr_schedule == 'none':
        ret_scheduler = LambdaLR(optimizer, lr_lambda=lambda x: 1)
    
    if opts.lr_schedule == 'plateau':
        def schedule_fn(metric):
            ret_scheduler.step(metric)
    else:
        def schedule_fn(metric):
            ret_scheduler.step()
    
    def schedule_step(metric=None):
        current_lr = optimizer.param_groups[0]['lr']
        schedule_fn(metric)
        new_lr = optimizer.param_groups[0]['lr']
        if new_lr != current_lr:
            name = subset['optimizer'] if isinstance(subset, dict) else subset
            name = ' ({0})'.format(name) if name else ''
            print('Learning rate{0} changed from {1:.2e} to {2:.2e}'.format(name, current_lr, new_lr))


    return schedule_step