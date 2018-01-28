import sys
import time

def barit(iterable, start=None, end=None, time_it=True, length=20, leave=True, filler='='):
    """Minimal progress bar for iterables.

    Args:
        iterable (list or tuple etc): An iterable.
        start (str, optional): String to place infront of the progress bar (default None).
        end (str, optional): String to add at the end of the progress bar (default None).
        timeit (bool, optional): Print elapsed time and ETA (default True).
        length (int, optional): Length of the progress bar not including ends (default 20).
        leave (bool, optional): If False, it deletes the progress bar once it ends (default True).
        filler (str, optional): Filler character for the progress bar (default '='). 

    Example:
        >>> for _ in dlt.util.barit(range(100), start='Count'):
        >>>     time.sleep(0.02)
        Count: [====================] 100.0%, (10/10), Total: 0.2s, ETA: 0.0s
  
    Note:
        barit can be put to silent mode using:

            >>> dlt.util.silent = True
   
     For a progress bar with more functionality have a look at tqdm_.
        
        .. _tqdm: https://github.com/tqdm/tqdm

    """
    if barit.silent:
        for x in iterable:
            yield x
    else:
        n_iter = len(iterable)
        start = '' if start is None else '{0}: '.format(start)
        bar = '[{0:' + str(length) + 's}]'
        perc = ' {0:.1f}%'
        counts = ', ({0}/{1})'.format('{0}', n_iter)
        time_est = ', ETA: {0:.1f}s' if time_it else ''
        elapsed_est = ', Total: {0:.1f}s' if time_it else ''
        end = '' if end is None else ', {0}'.format(end)

        bar_tmpl = '\r'+ start + '{bar}{perc}{counts}{elapsed_est}{time_est}' + end
        start_time = time.time()

        def print_bar(i, last_size):
            elapsed = (time.time()-start_time)
            full_bar = bar_tmpl.format(
                bar=bar.format(filler*int(i*length/n_iter)), 
                perc=perc.format(i*100/n_iter),
                counts=counts.format(i,n_iter),
                elapsed_est=elapsed_est.format(elapsed),
                time_est= time_est.format(elapsed*(n_iter-i)/i) if time_it and i>0 else '')
            full_bar += ' '*(max(last_size - len(full_bar), 0))
            sys.stdout.write(full_bar)
            sys.stdout.flush()
            return len(full_bar)
        
        last_len = print_bar(0, 0)
        for i_iter, x in enumerate(iterable, start=1):
            yield x
            last_len = print_bar(i_iter, last_len)

        sys.stdout.write('\n' if leave else '\r'+' '*last_len + '\r')
        sys.stdout.flush()

barit.silent = False