import sys
import logging

class CustomFileHandler(logging.FileHandler):
    def __init__(self, *args, **kwargs):
        super(CustomFileHandler, self).__init__(*args, **kwargs)
    def emit(self, record):
        try:
            msg = self.format(record)
            self.stream.write(msg)
        except Exception:
            self.handleError(record)

# From 
# https://stackoverflow.com/questions/1741972/how-to-use-different-formatters-with-the-same-logging-handler-in-python
class DispatchingFormatter:
    def __init__(self, formatters, default_formatter):
        self._formatters = formatters
        self._default_formatter = default_formatter

    def format(self, record):
        formatter = self._formatters.get(record.name, self._default_formatter)
        return formatter.format(record)

# Helper logger for outputting sys.stdout to a logfile
class DuplStdOut(object):
    def __init__(self, filename):
        self.stdout = sys.stdout
        
        self.logger = logging.getLogger('dlt_file_log')
        self.logger.setLevel(logging.INFO)
        self.logger.propagate = False
        file_handler = CustomFileHandler(filename)
        formatter = logging.Formatter('{message}', style='{')
        
        # Keep the formatting of the dlt logger
        file_handler.setFormatter(DispatchingFormatter({
                'dlt_file_log': formatter,
                'dlt': logging.getLogger('dlt').handlers[0].formatter
            },
            formatter,
        ))
        self.logger.addHandler(file_handler)
        
        # also add the file handler to dlt logger
        logging.getLogger('dlt').addHandler(file_handler)
        
        sys.stdout = self
 
    def __del__(self):
        if self.stdout is not None:
            sys.stdout = self.stdout
            self.stdout = None

    def write(self, msg):
        # if msg != os.linesep:
        self.logger.info(msg)
        self.stdout.write(msg)

    def flush(self):
        self.stdout.flush()
        for h in self.logger.handlers:
            h.flush()