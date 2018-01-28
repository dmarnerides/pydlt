from collections import OrderedDict

class Average(object):
    """Keeps an average of values."""
    def __init__(self):
        self.reset()

    def reset(self):
        """Resets the average to 0."""
        self.count = 0
        self.value = 0

    def add(self, value, count=1):
        """Adds a new value
        
        Args:
            value: Value to be added
            count (optional): Number of summed values that make the value given.
                Can be used to register multiple (summed) values at once (default 1).
        """
        self.count += count
        self.value += value

    def get(self):
        """Returns the current average"""
        return self.value/self.count

class Averages(object):
    """Keeps multiple named averages.
    
    Args:
        names(collection): Collection of strings to be used as names for the averages.
    """
    def __init__(self, names):
        self._names = names
        self.avgs = OrderedDict([(name, Average()) for name in names])

    def reset(self, names=None):
        """Resets averages to 0.

        Args:
            names (collection, optional): Collection of the names to be reset.
                If None is given, all the values are reset (default None). 
        """
        names = names or self._names
        for name in names:
            self.avgs[name].reset()

    def add(self, values, count=1):
        """Adds new values

        Args:
            values (dict or list): Collection of values to be added. 
                Could be given as a dict or a list. Order is preserved.
            count (int, optional): Number of summed values that make the total values given.
                Can be used to register multiple (summed) values at once (default 1).
        """
        if isinstance(values, dict):
            for name, value in values.items():
                if any([isinstance(value, x) for x in [list, tuple]]):
                    self.avgs[name].add(*value)
                else:
                    self.avgs[name].add(value, count)
        else:
            for i, value in enumerate(values):
                if any([isinstance(value, x) for x in [list, tuple]]):
                    self.avgs[i].add(*value)
                else:
                    self.avgs[i].add(value, count)

    def get(self, names=None, ret_dict=False):
        """Returns the current averages
        
        Args:
            names (str or list, optional): Names of averages to be returned.
            ret_dict (bool, optional): If true return the results in a dictionary,
                otherwise a list.

        Returns:
            dict or list: The averages.
        """
        if isinstance(names, str):
            if ret_dict:
                return {names: self.avgs[names].get()}
            else:
                return self.avgs[names].get()
        if names is None:
            names = self._names
        if ret_dict:
            return {name: self.avgs[name].get() for name in names}
        else:
            return [self.avgs[name].get() for name in names]

    def names(self):
        """Returns the names of the values held."""
        return self._names