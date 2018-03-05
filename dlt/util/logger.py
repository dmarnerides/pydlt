from os import path
import logging
from .paths import process
from .misc import is_tensor, is_variable, _get_scalar_value

class Logger(object):
    """Logs values in a csv file.

    Args:
        name (str): Filename without extension.
        fields (list or tuple): Field names (column headers).
        directory (str, optional): Directory to save file (default '.').
        delimiter (str, optional): Delimiter for values (default ',').
        resume (bool, optional): If True it appends to an already existing
            file (default True).

    """
    def __init__(self, name, fields, directory=".",
                 delimiter=',', resume=True):
        self.filename = name + ".csv"
        self.directory = process(path.join(directory), True)
        self.file = path.join(self.directory, self.filename)
        self.fields = fields
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.INFO)
        # Write header
        if not resume or not path.exists(self.file):
            with open(self.file, 'w') as f:
                f.write(delimiter.join(self.fields)+ '\n')

        file_handler = logging.FileHandler(self.file)
        field_tmpl = delimiter.join(['%({0})s'.format(x) for x in self.fields])
        file_handler.setFormatter(logging.Formatter(field_tmpl))
        self.logger.addHandler(file_handler)

    def __call__(self, values):
        """Same as :meth:`log`"""
        self.log(values)

    def log(self, values):
        """Logs a row of values.

        Args:
            values (dict): Dictionary containing the names and values.
        """
        for key, val in values.items():
            if is_variable(val):
                values[key] = _get_scalar_value(val.data)
            if is_tensor(val):
                values[key] = _get_scalar_value(val)
        self.logger.info('', extra=values)