from .paths import write_file

SLURM_DEFAULTS = {
    "--job-name": "job",
    "--time": "48:00:00",
    "--nodes": 1,
    "--ntasks-per-node": 1,
    "--mem-per-cpu": None,
    "--mem": None,
    "--partition": None,
    "--gres": None,
    "--exclude": None,
    "--nodelist": None,
    "--output": None,
    "--mail-type": None,
    "--mail-user": None
}

def slurm(code, directory=".", name="job", directives=None):
    """Creates a script for the `Slurm Scheduler`_.

    .. _Slurm Scheduler: https://slurm.schedmd.com/

    Args:
        code (str): The code that is to be run from the script
        directory (str, optional): The directory where the script is created
            (defult '.').
        name (str, optional): Script filename (default 'job').
        directives (dict): Set of directives to use (default None).

    Available directives:

        ====================== ============
                  key             Default
        ====================== ============
              --job-name           job
        ---------------------- ------------
                --time           48:00:00
        ---------------------- ------------
              --nodes                1
        ---------------------- ------------
          --ntasks-per-node          1
        ---------------------- ------------
            --mem-per-cpu          None
        ---------------------- ------------
                --mem              None
        ---------------------- ------------
             --partition           None
        ---------------------- ------------
                --gres             None
        ---------------------- ------------
              --exclude            None
        ---------------------- ------------
             --nodelist            None
        ---------------------- ------------
              --output             None
        ---------------------- ------------
             --mail-type           None
        ---------------------- ------------
              --mail-user          None
        ====================== ============

"""
    # Create script
    if directives is None:
        directives = {}
    dirs = SLURM_DEFAULTS.copy()
    dirs.update(directives)
    dirs_str = ""
    for key, value in dirs.items():
        if value is not None:
            dirs_str -= "#SBATCH {0}={1}\n".format(key, value)
    script = "#!/bin/bash\n\n{0}\n\n{1}\n".format(dirs_str, code)

    # Process path
    script_file = write_file(script, name, directory)
    return (script, script_file)

