from ..util import Checkpointer, count_parameters
from .opts import fetch_opts


def model_checkpointer(model, preprocess=None, subset=None):
    """Returns the model checkpointer. Configurable using command line arguments.
    
    The function also loads any previous checkpoints if present.

    Args:
        model (nn.Module): The network for the checkpointer.
        preprocess (callable, optional): Callable to change the loaded state dict before
            assigning it to the network.
        subset (string, optional): Specifies the subset of the relevant
            categories, if any of them was split (default, None).

    Relevant Command Line Arguments:

        - **general**: `--experiment_name`, `--save_path`.
        - **model**: `--overwrite_model_chkp`, `--timestamp_model_chkp`, `--count_model_chkp`.

    Note:
        Settings are automatically acquired from a call to :func:`dlt.config.parse`
        from the built-in ones. If :func:`dlt.config.parse` was not called in the 
        main script, this function will call it.
    """

    opts = fetch_opts(['general', 'model'], subset)
    name = subset['model'] if isinstance(subset, dict) else subset
    exp_name = opts.experiment_name + '_' if opts.experiment_name != '' else ''
    model_chkp = Checkpointer('{0}{1}weights'.format(exp_name, '' if name is None else name + '_'),
                                  directory=opts.save_path, overwrite=opts.overwrite_model_chkp,
                                  timestamp=opts.timestamp_model_chkp, add_count=opts.count_model_chkp)
    model_chkp.load(model, preprocess=preprocess)
    print('{0}\n{1}\n{0}\n{2}\nParameters: {3}\n{0}'.format('-'*80, name.capitalize() if name is not None else 'Model', model, count_parameters(model)))
    return model_chkp
