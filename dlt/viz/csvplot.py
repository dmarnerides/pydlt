import os
import time
import argparse
from itertools import compress
from collections import OrderedDict
import matplotlib
from matplotlib.animation import FuncAnimation
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from ..util import str2bool, compose, str_is_int, is_array, sub_avg, moving_avg, sub_var, moving_var


def _read_csv(f, **kargs):
    if os.path.isfile(f):
        df = pd.read_csv(f, **kargs) 
    else:
        df = pd.DataFrame()
    return df

class DataHandler(object):
    def __init__(self, file_lists, header, ignore_duplicates, transform, names):
        self.header = 0 if header else None
        self.ignore_duplicates = ignore_duplicates
        self.transform = transform
        self.filenames = [f[0] for f in file_lists]
        self.orig_columns = [f[1:] if len(f)>1 else [0] for f in file_lists]
        self.orig_columns = [ [int(c) if str_is_int(c) else c for c in j] for j in self.orig_columns ]
        self._group_names(names)

    def _group_names(self, names):
        if names is None:
            self.orig_names = None
            return
        count = 0
        self.orig_names = []
        for c in self.orig_columns:
            self.orig_names.append([])
            for _ in c:
                if count == len(names):
                    self.orig_names[-1].append('no_name')
                else:
                    self.orig_names[-1].append(names[count])
                    count += 1

    #TODO: MAKE THIS MORE EFFICIENT AND LESS OBFUSCATED
    def get(self):
        dfs, names, columns = [], [], []
        for i_file, fname in enumerate(self.filenames):
            if os.path.isfile(fname):
                dfs.append(_read_csv(fname, header=self.header))
                columns.append([])
                for i_col, c in enumerate(self.orig_columns[i_file]):
                    if isinstance(c, int) and len(dfs[-1].columns) > c:
                        columns[-1].append(dfs[-1].columns[c])
                        if self.orig_names is not None:
                            names.append(self.orig_names[i_file][i_col])
                    elif c in dfs[-1]:
                        columns[-1].append(c)
                        if self.orig_names is not None:
                            names.append(self.orig_names[i_file][i_col])
        # concatenate all the dataframes into a single one
        if not dfs:
            return None
        data = pd.concat([pd.DataFrame(df.loc[:,columns[i]]) for i,df in enumerate(dfs)], axis=1)
        if not self.ignore_duplicates:
            cols = pd.Series(data.columns)
            for dup in data.columns.get_duplicates():
                cols[data.columns.get_loc(dup)]=['{0}-dupl{1}'.format(dup,d_idx+1) for d_idx in range(np.sum(dup == cols))]
            data.columns = cols
        if self.orig_names is not None:
            data.columns = names
        return self.transform(data) if len(data.index) > 0 else data

    __call__ = get


class CustomAction(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        if not '_ordered_args' in namespace:
            setattr(namespace, '_ordered_args', [])
        previous = namespace._ordered_args
        previous.append(self.dest)
        setattr(namespace, self.dest, values)
        setattr(namespace, '_ordered_args', previous)


def _get_opts(use_args=None):
    parser = argparse.ArgumentParser(description='Easily plot from csvs.')
    arg = parser.add_argument
    arg('-f','--file', required=True, nargs='+', help='File followed by columns (names or indices). If no column, defaults to first one (0).', action='append')
    
    ## Plot arguments
    arg('-t', '--title', nargs='+', help='Title(s) for (sub)plot', default=None, action=CustomAction)
    arg('-x', '--xlabel', nargs='+', help='x label(s).', default=None, action=CustomAction)
    arg('-y', '--ylabel', nargs='+', help='y label(s).', default=None, action=CustomAction)
    arg('--logx', type=str2bool, help='Logarithmic x.', default=False, action=CustomAction)
    arg('--logy', type=str2bool, help='Logarithmic y.', default=False, action=CustomAction)
    arg('--loglog', type=str2bool, help='Use log scaling on both x and y axes.', default=False, action=CustomAction)
    arg('--kind', choices=['line', 'bar', 'barh', 'hist', 'box', 'kde', 'density', 'area', 'pie', 'scatter', 'hexbin'], default='line', action=CustomAction)
    arg('--subplots', type=str2bool, help='Make separate subplots for each column.', default=False, action=CustomAction)
    arg('--layout', nargs=2, help='rows columns for the layout of subplots.', default=None, action=CustomAction)
    arg('--use_index', type=str2bool, help='Use index as ticks for x axis.', default=False, action=CustomAction)
    arg('--sharex', type=str2bool, help='In case subplots=True, share x axis.', default=False, action=CustomAction)
    arg('--sharey', type=str2bool, help='In case subplots=True, share y axis.', default=False, action=CustomAction)
    arg('--figsize', nargs=2, help='Figure size (width and height) in inches.', default=None, action=CustomAction)
    arg('--grid', type=str2bool, help='Axis grid lines.', default=None, action=CustomAction)
    arg('--legend', type=lambda x: x if x=='reverse' else str2bool(x), help='Place legend on axis subplots.', default=True, action=CustomAction)
    arg('--names', nargs='+', help='Names for each plotted series.', default=None, action=CustomAction)
    arg('--style', nargs='+', help='matplotlib line style per column.', action=CustomAction)
    arg('--rot', type=int, help='Rotation for ticks.', default=None, action=CustomAction)
    arg('--xlim', type=int, nargs=2, help='X axis limits.', default=None, action=CustomAction)
    arg('--ylim', type=int, nargs=2, help='Y axis limits.', default=None, action=CustomAction)
    arg('--fontsize', type=int, help='Font size for xticks and yticks.', default=None, action=CustomAction)
    arg('--colormap', help='Colormap to select colors from.', default=None, action=CustomAction)
    arg('--position', type=float, help='Specify relative alignments for bar plot layout. [0-1]', default=0.5, action=CustomAction)
    arg('--table', type=str2bool, help='If True, draw a table.', default=False, action=CustomAction)
    ## Transformations
    arg('-a', '-sa', '--sub_avg', nargs='+', type=int, help='Plot in averages', default=None, action=CustomAction)
    arg('-ma', '--mov_avg', nargs='+', type=int, help='Plot moving average', default=None, action=CustomAction)
    arg('-v', '-sv', '--sub_var', nargs='+', type=int, help='Plot in variances', default=None, action=CustomAction)
    arg('-mv', '--mov_var', nargs='+', type=int, help='Plot moving variances', default=None, action=CustomAction)
    arg('-sh', '--head', nargs='+', type=int, help='Select head', default=None, action=CustomAction)
    arg('-st', '--tail', nargs='+', type=int, help='Select tail', default=None, action=CustomAction)
    arg('-rh', '--rhead', nargs='+', type=int, help='Remove head', default=None, action=CustomAction)
    arg('-rt', '--rtail', nargs='+', type=int, help='Remove tail', default=None, action=CustomAction)
    arg('-lv', '--logarithm', type=str2bool, help='Natural logarithm', default=None, action=CustomAction)
        ## Misc
    arg('--display', type=str2bool, help='Display plot.', default=True, action=CustomAction)
    arg('--header', type=str2bool, help='CSV has header.', default=True, action=CustomAction)
    arg('--ignore_duplicates', type=str2bool, help='Use only one of columns with the same name.', default=False, action=CustomAction)
    arg('-s', '--save', help='Filename to save figure (Only if --refresh is not set).', default='none', action=CustomAction)
    arg('-r', '--refresh', type=float, help='Time interval (ms) to refresh figure data from csv.', default=None, action=CustomAction)
    
    if use_args is not None:
        opt = parser.parse_args(use_args)
    else:
        opt = parser.parse_args()

    return opt


def plot_csv(use_args=None):
    r"""Plots data from csv files using pandas.
    
    Also usable as a command line program `dlt-plot`.

    Args:
        use_args (dict, optional): Arguments to use instead of (command line) args.

    Example:
        Use with command line:

        .. code-block:: console

            $ dlt-plot -f training.csv --sub_avg 500

        From inside a script:
            
            >>> dlt.viz.plot_csv(['--file', 'training.csv', '--sub_avg', '500'])

    Note:
        For information on available functionality use:

        .. code-block:: console

            $ dlt-plot --help
    """
    opt = _get_opts(use_args)
    # make Namespace a dict
    opt_dict = vars(opt)
    # Remove size one arrays
    for v in ['xlabel', 'ylabel', 'title', 'head', 'tail', 'rhead', 'rtail', 
              'sub_avg', 'mov_avg', 'sub_var', 'mov_var', 'logarithm']:
        if opt_dict[v] is not None and len(opt_dict[v]) == 1:
            opt_dict[v] = opt_dict[v][0]
    
    func_dict = OrderedDict([
        ('sub_avg', (lambda x, v: sub_avg(x, v))),
        ('mov_avg', (lambda x, v: moving_avg(x, v))),
        ('sub_var', (lambda x, v: sub_var(x, v))),
        ('mov_var', (lambda x, v: moving_var(x, v))),
        ('head', (lambda x, v: x[:v])),
        ('tail', (lambda x, v: x[-v:])),
        ('rhead', (lambda x, v: x[v:])),
        ('rtail', (lambda x, v: x[:-v])),
        ('logarithm', (lambda x, v: np.log(x)))
    ])
    funcs = []
    if '_ordered_args' in opt_dict:
        for key in opt_dict['_ordered_args']:
            if key in func_dict and opt_dict[key] is not None:
                if isinstance(opt_dict[key], list):
                    func = (lambda x,i,k=key: func_dict[k](x,opt_dict[k][i]))
                else:
                    func = (lambda x,i,k=key: func_dict[k](x,opt_dict[k]))
                funcs.append(lambda df, f=func: pd.DataFrame(OrderedDict([(col, pd.Series(f(df[col], i))) for i, col in enumerate(df.columns) ])))

    transform = compose(funcs)

    plot_args_list = ['logx', 'logy', 'loglog', 'kind', 'subplots', 'layout',
                      'use_index', 'sharex', 'sharey', 'figsize', 'grid', 'legend',
                      'style', 'rot', 'xlim', 'ylim', 'fontsize', 'colormap',
                      'title', 'table']
    plot_args = {k: v for k, v in opt_dict.items() if k in plot_args_list}
    
    data_handler = DataHandler(opt.file, opt.header, opt.ignore_duplicates, transform, opt.names)

    def _update(frame=None):
        data = data_handler()
        axes.clear()
        _set_labels()
        if data is not None and len(data.index) > 0:
            try:
                return data.plot(ax = axes, **plot_args).get_lines()
            except:
                return axes.plot([],[])
        else:
            return axes.plot([],[])

    def _set_labels():
        if is_array(axes):
            for i,a in enumerate(axes):
                a.set_xlabel(opt_dict['xlabel'][i] if isinstance(opt_dict['xlabel'], list) else opt_dict['xlabel'] if opt_dict['xlabel'] is not None else 'x') 
                a.set_ylabel(opt_dict['ylabel'][i] if isinstance(opt_dict['ylabel'], list) else opt_dict['ylabel'] if opt_dict['ylabel'] is not None else 'y')
        else:
            axes.set_xlabel(opt_dict['xlabel'] if opt_dict['xlabel'] is not None else 'x')
            axes.set_ylabel(opt_dict['ylabel'] if opt_dict['ylabel'] is not None else 'y')

    figure, axes = plt.subplots(num='viz')
    _update()
    if opt.refresh is not None and opt.refresh > 0.001:
        anim = FuncAnimation(fig=figure, func=_update, frames=None,
                             init_func=None, save_count=None, 
                             interval=opt.refresh*1000,
                             repeat=False, blit=False)
    else:
        if opt.save != 'none':
            plt.savefig(opt.save, bbox_inches='tight')

    plt.show()