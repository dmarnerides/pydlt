from ..util import Checkpointer
from .opts import fetch_opts


def trainer_checkpointer(trainer, subset=None):
    """Returns the trainer checkpointer. Configurable using command line arguments.
    
    The function also loads any previous checkpoints if present.

    Args:
        trainer (BaseTrainer): The trainer for the checkpointer.
        subset (string, optional): Specifies the subset of the relevant
            categories, if any of them was split (default, None).

    Relevant Command Line Arguments:

        - **general**: `--experiment_name`, `--save_path`.
        - **trainer**: `--overwrite_trainer_chkp`, `--timestamp_trainer_chkp`, `--count_trainer_chkp`.

    Note:
        Settings are automatically acquired from a call to :func:`dlt.config.parse`
        from the built-in ones. If :func:`dlt.config.parse` was not called in the 
        main script, this function will call it.
    """

    opts = fetch_opts(['general', 'trainer'], subset)
    name = subset['trainer'] if isinstance(subset, dict) else subset
    exp_name = opts.experiment_name + '_' if opts.experiment_name != '' else ''
    trainer_chkp = Checkpointer('{0}{1}trainer'.format(exp_name, '' if name is None else name + '_'),
                                  directory=opts.save_path, overwrite=opts.overwrite_trainer_chkp,
                                  timestamp=opts.timestamp_trainer_chkp, add_count=opts.count_trainer_chkp)
    trainer_chkp.load(trainer)
    return trainer_chkp
