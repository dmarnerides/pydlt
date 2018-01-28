import os
from os import path
import argparse
from .paths import process, copy_to_dir, write_file
from .slurm import slurm

def read_argument_file(file_path):
    file_path = process(file_path)
    ret = {}
    with open(file_path, 'r') as f:
        for line in f:
            split = line.strip().split(' ')
            ret[split[0]] = split[1]
    return ret


# TODO: Keep directory structure of extras copied to directory.
def dispatch():
    r"""Creates a self contained experiment in a directory
    
    Also usable as a command line program `dlt-dispatch`.

    Example:
        Use with command line:

        .. code-block:: console

            $ dlt-dispatch test_low_lr -d ~/experiments -m main.py -e models.py data.py -c settings_low_lr.cfg

    Note:
        For information on available functionality use:

        .. code-block:: console

            $ dlt-dispatch --help
    """
    ### Parse arguments
    parser = argparse.ArgumentParser(description='Create a self contained experiment for PyTorch.')
    arg = parser.add_argument
    arg('name', help='Name of experiment')
    arg('-d', '--directory', default='~/experiments', help='Parent directory of experiment.')
    arg('-m', '--main', default='main.py', help='Main file')
    arg('-e', '--extras', nargs='+', default=None, help='Other files to be copied (e.g. model.py)')
    arg('-c', '--config', default=None, help='File with experiment configuration')
    arg('-s', '--slurm_config', default=None, help='File with slurm configuration')
    arg('-p', '--slurm_pre_code', default=None, help='File with slurm code to be ran before the main script.')
    opt = parser.parse_args()

    directory = process(path.join(opt.directory, opt.name), create=True)
    if not path.isfile(opt.main):
        exit("Could not find script " + opt.main)
    # Copy the script and config
    copy_to_dir(opt.main, directory)
    if opt.config is not None:
        copy_to_dir(opt.config, directory)
    # Copy extras
    if opt.extras is not None:
        for f in opt.extras:
            copy_to_dir(f, directory)
    # Create run file
    if opt.config is not None:
        command = "python {0} $(< {1})\n".format(
            path.basename(opt.main), path.basename(opt.config))
    else:
        command = "python {0}\n".format(path.basename(opt.main))

    run_filename = write_file("#!/bin/bash\n\n" + command, 'run.sh', directory)
    os.chmod(run_filename, 0o775)
    # Create slurm
    if opt.slurm_config is not None:
        slurm_settings = read_argument_file(opt.slurm_config)
        if opt.slurm_pre_code is not None:
            command = opt.slurm_pre_code + "\n\n" + command
        slurm_settings["--job-name"] = opt.name
        slurm(command, directory, directives=slurm_settings)