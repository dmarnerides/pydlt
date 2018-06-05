import os
import sys
import argparse
import logging
from functools import partial
from ..util import paths
from ..util.paths import process
from ..util import str2bool
from .helpers import DuplStdOut

# This is to allow commented lines in the configuration files
def _convert_arg_line_to_args(line):
    for arg in line.split():
        if not arg.strip():
            continue
        if arg[0] == '#':
            break
        yield arg

def print_opts(opt):
    """Prints the parsed command line options."""
    size = 80
    title = 'Settings:'
    if hasattr(opt, '__dict__'):
        opt = opt.__dict__
    bar = '-'*size
    print('{0}\n  {1}\n{0}'.format(bar, title))
    for k, v in opt.items():
        print('  {0}: {1}'.format(k, v))
    print(bar)

def make_subsets(subsets):
    """Splits command line argument categories into subsets.
    
    The subset names are appended at the end of each of the categories options
    after an underscore. For example the dataset category can be split into 
    training and validation subsets by passing the following as `subsets`:
    
    .. code-block:: python3

        subsets = {dataset=['training', 'validation']}
    
    This will cause::

        - `--data` to split into `--data_training` and `--data_validation`.
        - `--load_all` to split into `--load_all_training` and `--load_all_validation`
        - etc ...

    Args:
        subsets (dict, optional): Dictionary containing parameter categories as
            keys and its subsets as a list of strings (default None).
    """
    parse.subsets = subsets

def add_extras(extras):
    """Adds extra options for the parser, in addition to the built-in ones.

    Args:
        extras (list or dict, optional): Extra command line arguments to parse.
            If a list is given, a new category `extras` is added with the
            listed arguments. If a dict is given, the keys contain the category
            names and the elements are lists of arguments (default None).

    Note:
        The `extras` parameter must have one of the following structures:

        .. code-block:: python3

            # List of arguments as dicts (this will add extras to )
            extras = [dict(flag='--arg1', type=float),
                      dict(flag='--arg2', type=int),
                      dict(flag='--other_arg', type=int)]
            # OR dictionary with category names in keys 
            # and lists of dicts for as values for each category
            extras = {'my_category': [dict(flag='--arg1', type=float),
                                      dict(flag='--arg2', type=int)]
                      'other_category': [dict(flag='--other_arg', type=int)]}

        The keys accepted in the argument dictionaries are the ones used as
        arguments in the argparse_ package `add_argument` function.

        .. _argparse: https://docs.python.org/3/library/argparse.html
DuplStdOut
    Warning:
        The parser takes strings as inputs, so passing 'False' at the command
        for a bool argument, it will be converted to *True*. Instead of using
        type=bool, use type=dlt.util.str2bool.

    """
    if isinstance(extras, dict):
        for x in extras:
            if x in parse.param_dict.keys():
                print('WARNING: Parameter category \'{0}\' overwritten'.format(x))
            for fl in ['flag', 'flags']:
                for extra_args in extras[x]:
                    if fl in extra_args and isinstance(extra_args[fl], str):
                        extra_args[fl] = [extra_args[fl]]

        parse.param_dict.update(extras)
    elif isinstance(extras, list):
        parse.param_dict['extras'] = extras

def parse(verbose=False):
    """Parses Command Line Arguments using the built-in settings (and any added extra).

    Args:
        verbose (bool, optional): Print the parsed settings (default False).

    Comes with built-in commonly used settings. To add extra settings use
    :func:`add_extras`.

    For a comprehensive list of available settings, their categories and
    use of configuration files please see the :ref:`config-example` example.
    """

    if not hasattr(parse, 'subsets'):
        parse.subsets = {}

    parser = argparse.ArgumentParser(fromfile_prefix_chars='@')
    parser.convert_arg_line_to_args = _convert_arg_line_to_args
    arg = parser.add_argument

    # Go through flags and make ones that are strings into lists:
    for category, params in parse.param_dict.items():
        for p in params:
            for fl in ['flag', 'flags']:
                if fl in p and isinstance(p[fl], str):
                    p[fl] = [p[fl]]

    # This looks bad but at least it works.
    # It goes through the categories and retrieves the settings from the
    # dictionaries in the list of each category and passes them to parser.add_argument
    # If the category needs to be subset, the '_<subset_name>' is added for each option
    # for each subset name (along with an appended string at the end of the help 
    # setting if there is one. 
    for category, params in parse.param_dict.items():
        if category in parse.subsets:
            for subset in parse.subsets[category]:
                for p in params:
                    
                    if 'help' in p:
                        help_str = p['help'] + ' ({0})'.format(subset)
                    
                    arg(*[y + '_{0}'.format(subset)
                            for x in [p[fl]
                                        for fl in ['flag', 'flags']
                                            if fl in p]
                                for y in x],
                         **{key: val
                                for key, val in p.items()
                                    if key not in ['flag', 'flags', 'help']},
                         help = help_str if 'help' in p else None)
        else:
            for p in params:
                arg(*[y for fl in ['flag', 'flags'] 
                        if fl in p for y in p[fl]], 
                    **{key: val for key, val in p.items() 
                        if key not in ['flag', 'flags']})
    opt = parser.parse_args()
    if opt.experiment_name != '':
        opt.save_path = os.path.join(opt.save_path, opt.experiment_name)
    
    paths.make(opt.save_path)
    
    # Create an event log file
    if opt.create_log:
        logfile = os.path.join(opt.save_path, 'dlt.out.log')
        DuplStdOut(logfile)

    # Print all arguments 

    if verbose:
        print_opts(opt)
    parse.opt = opt
    return opt

class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self

def fetch_opts(categories, subset=None):
    if not hasattr(parse, 'opt'):
        parse()
    opt = parse.opt
    if isinstance(categories, str):
        categories = [categories]
    ret = AttrDict()
    for category in categories:
        if isinstance(subset, dict):
            subset = subset[category]
        elif isinstance(subset, str):
            subset = subset
        else:
            subset = None
        postfix = '_{0}'.format(subset) if subset is not None and subset != '' else ''
        names = parse.default_avail_params[category]
        current = AttrDict({name: getattr(opt, name + postfix) if hasattr(opt, name + postfix) else getattr(opt, name) for name in names})
        ret.update(current)
    return ret

parse.torchvision_datasets = ['mnist', 'fashionmnist', 'cifar10', 'cifar100']

parse.optimizers = ['adam', 'sgd', 'adadelta', 'adagrad', 'sparseadam', 'adamax', 'rmsprop']

parse.param_dict = {
    'dlt': [dict(flags=['--create_log'], type=str2bool, default=True, help='Output all std_out to a log file.'),
            ],
    'general': [dict(flags=['--experiment_name'], default='', help='Name of experiment'),
                 dict(flags=['--save_path'], type=partial(process, create=True), default='.', help='Root directory for experiments'),
                 dict(flags=['--seed'], type=int, default=None, help='Seed for random number generation.'),
                 dict(flags=['--max_epochs'], type=int, default=100000, help='Maximum number of epochs')],
    'dataset': [ dict(flags=['--data'], type=process, default='.', help='Data directory' ),
                 dict(flags=['--load_all'], type=str2bool, default=False, help='Load data in memory'),
                 dict(flags=['--torchvision_dataset'],type = str.lower, choices=parse.torchvision_datasets, default=None, help='Specific dataset to use'),
                 dict(flags=['--extensions'], nargs='+', default=['jpg'], help='Extensions of data to load.')],
    'dataloader': [ dict(flags=['--num_threads'], type=int, default=4, help='Number of data loading threads'),
                     dict(flags=['--batch_size'], type=int, default=1, help='Batch size for loader'),
                     dict(flags=['--shuffle'], type=str2bool, default=True, help='Loader shuffles data each epoch'),
                     dict(flags=['--pin_memory'], type=str2bool, default=True, help='Pin tensor memory for efficient GPU loading'),
                     dict(flags=['--drop_last'], type=str2bool, default=False, help='Drop last batch if its size is less than batch size')],
    'model': [ dict(flags=['--overwrite_model_chkp'], type=str2bool, default=True, help='Overwrite model checkpoints'),
               dict(flags=['--timestamp_model_chkp'], type=str2bool, default=False, help='Add timestamp to model checkpoints'),
               dict(flags=['--count_model_chkp'], type=str2bool, default=True, help='Add count to model checkpoints') ],
    'optimizer': [ dict(flags=['--optimizer'],type = str.lower, choices = parse.optimizers, default='adam', help='Optimizer'),
                   dict(flags=['--overwrite_optimizer_chkp'], type=str2bool, default=True, help='Overwrite optimizer checkpoints'),
                   dict(flags=['--timestamp_optimizer_chkp'], type=str2bool, default=False, help='Add timestamp to optimizer checkpoints'),
                   dict(flags=['--count_optimizer_chkp'], type=str2bool, default=True, help='Add count to optimizer checkpoints'),
                   dict(flags=['--lr'], type=float, default=1e-3, help='Learning rate'),
                   dict(flags=['--momentum'], type=float, default=0.9, help='SGD Momentum'),
                   dict(flags=['--dampening'], type=float, default=0.0, help='SGD Dampening'),
                   dict(flags=['--beta1'], type=float, default=0.9, help='Adam beta1 parameter'),
                   dict(flags=['--beta2'], type=float, default=0.99, help='Adam beta2 parameter'),
                   dict(flags=['--rho'], type=float, default=0.9, help='Adadelta rho parameter'),
                     dict(flags=['--alpha'], type=float, default=0.99, help='RMSprop alpha parameter'),
                     dict(flags=['--centered'], type=str2bool, default=False, help='RMSprop centered flag'),
                     dict(flags=['--lr_decay'], type=float, default=0.0, help='Adagrad lr_decay'),
                     dict(flags=['--optim_eps'], type=float, default=1e-8, help='Term added to denominator for numerical stability.'),
                     dict(flags=['--weight_decay'], type=float, default=0.0, help='Weight decay / L2 regularization.')],
    'scheduler': [ dict(flags=['--lr_schedule'], choices=['plateau', 'step', 'none'], default='step', help='Learning rate schedule'),
                     dict(flags=['--lr_step_size'], type=int, default=100, help='Epochs per learning rate decrease (step).'),
                     dict(flags=['--lr_patience'], type=int, default=10, help='Epochs of patience for metric (plateau).'),
                     dict(flags=['--lr_cooldown'], type=int, default=0, help='Epochs of cooldown period after lr change (plateau).'),
                     dict(flags=['--lr_min'], type=float, default=1e-7, help='Minimum learning rate (plateau)'),
                     dict(flags=['--lr_ratio'], type=float, default=0.5, help='Ratio to decrease learning rate by (all)')],
    'gpu': [dict(flags=['--use_gpu'], type=str2bool, default=True, help='Use GPU'),
            dict(flags=['--device'], type=int, default=0, help='GPU device ID'),
            dict(flags=['--cudnn_benchmark'], type=str2bool, default=True, help='Use cudnn benchmark mode')],
    'trainer': [ dict(flags=['--overwrite_trainer_chkp'], type=str2bool, default=True, help='Overwrite trainer checkpoints'),
               dict(flags=['--timestamp_trainer_chkp'], type=str2bool, default=False, help='Add timestamp to trainer checkpoints'),
               dict(flags=['--count_trainer_chkp'], type=str2bool, default=True, help='Add count to trainer checkpoints') ],
}

parse.default_avail_params = {category: [param['flags'][0][2:] for param in params] 
                        for category, params in parse.param_dict.items()}