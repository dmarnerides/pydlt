import os
import matplotlib.pyplot as plt
import cv2
import torch
import pandas as pd
from ..util.paths import process
from ..util import torch2cv, map_range, make_grid

def _get_tensor(x):
    x = x[0] if torch.typename(x) in ['tuple', 'list'] else x
    x = x.data if hasattr(x,'data') else x
    return x

def save_image(img, title):
    to_save = ((img / img.max())*255).astype(int)
    cv2.imwrite('{0}.png'.format(title), to_save)


def process_none(x):
    if x is None:
        x = []
    elif not any((isinstance(x, y) for y in [list, tuple])):
        x = [x]
    return x

def _register(net, hook, modules=None, match_names=None, do_forward=True):
    modules = process_none(modules)
    match_names = process_none(match_names)
    for mod_name, mod in net.named_modules():
        name_match = any([torch.typename(modules).find(x) >= 0 for x in match_names])
        instance_match = any([isinstance(mod, x) for x in modules])
        if instance_match or name_match:
            if do_forward:
                mod.register_forward_hook(hook(mod_name))
            else:
                mod.register_backward_hook(hook(mod_name))
    return net

def _hook_generator(do_input=False, do_output=True, tag='', save_path='.', replace=True, histogram=True, bins=100, mode='forward', param_names=None):
    save_path = process(save_path, True)
    tensor_names = ['input', 'output'] if mode in ['forward', 'parameters'] else ['grad_input', 'grad_output']    
    def get_hook(module_name):
        counter = 1
        def hook(module, inp=None, out=None):
            nonlocal counter, tensor_names
            if mode == 'parameters':
                tensors = {x: _get_tensor(getattr(module, x)) for x in param_names}
            else:
                tensors = [(tensor_names[0], inp, do_input), (tensor_names[1], out, do_output)]
                tensors = {x[0]: _get_tensor(x[1]) for x in tensors if x[2]}
            for tensor_name, data in tensors.items():
                if data is None:
                    continue
                title_end = '' if replace else '-{0:06d}'.format(counter) 
                title_end = title_end + '-hist' if histogram else title_end
                title = '{0}-{1}-{2}{3}'.format(tag, module_name, tensor_name, title_end)
                if histogram:
                    img = torch2cv(data)
                    df = pd.DataFrame(img.reshape(img.size))
                    fig, ax = plt.subplots()
                    df.hist(bins=bins, ax=ax)
                    fig.savefig(os.path.join(save_path, '{0}.png'.format(title)))
                    plt.close(fig)
                else:
                    if data.dim() > 1:
                        img = torch2cv(make_grid(data, color=False))
                        to_save = (map_range(img)*255).astype(int)
                        cv2.imwrite(os.path.join(save_path, '{0}.png'.format(title)), to_save)
            counter = counter+1
        return hook
    return get_hook



def forward_hook(net, modules=None, match_names=None, do_input=False, do_output=True, 
                       tag='', save_path='.', replace=True, histogram=True, bins=100):
    """Registers a forward hook to a network's modules for vizualization of the inputs and outputs.

    When net.forward() is called, the hook saves an image grid or a histogram 
    of input/output of the specified modules.

    Args:
        net (nn.Module): The network whose modules are to be visualized.
        modules (list or tuple, optional): List of class definitions for the
            modules where the hook is attached e.g. nn.Conv2d  (default None).
        match_names (list or tuple, optional): List of strings. If any modules
            contain one of the strings then the hook is attached (default None).
        do_input (bool, optional): If True the input of the module is 
            visualized (default False).
        do_output (bool, optional): If True the output of the module is 
            visualized (default True).
        tag (str, optional): String tag to attach to saved images (default None).
        save_path (str, optional): Path to save visualisation results 
            (default '.').
        replace (bool, optional): If True, the images (from the same module) 
            are replaced whenever the hook is called (default True).
        histogram (bool, optional): If True then the visualization is a
            histrogram, otherwise it's an image grid.
        bins (bool, optional): Number of bins for histogram, if `histogram` is
            True (default 100).

    Note:
        * If modules or match_names are not provided then no hooks will be
          attached.
    """
    hook = _hook_generator(do_input,do_output,tag,save_path,replace, histogram, bins, 'forward')
    _register(net, hook, modules, match_names, True)
    return net

def backward_hook(net, modules=None, match_names=None, do_grad_input=False, do_grad_output=True, 
                       tag='', save_path='.', replace=True, histogram=True, bins=100):
    """Registers a backward hook to a network's modules for vizualization of the gradients.

    When net.backward() is called, the hook saves an image grid or a histogram 
    of grad_input/grad_output of the specified modules.

    Args:
        net (nn.Module): The network whose gradients are to be visualized.
        modules (list or tuple, optional): List of class definitions for the
            modules where the hook is attached e.g. nn.Conv2d  (default None).
        match_names (list or tuple, optional): List of strings. If any modules
            contain one of the strings then the hook is attached (default None).
        do_grad_input (bool, optional): If True the grad_input of the module is 
            visualized (default False).
        do_grad_output (bool, optional): If True the grad_output of the module 
            is visualized (default True).
        tag (str, optional): String tag to attach to saved images (default None).
        save_path (str, optional): Path to save visualisation results 
            (default '.').
        replace (bool, optional): If True, the images (from the same module) 
            are replaced whenever the hook is called (default True).
        histogram (bool, optional): If True then the visualization is a
            histrogram, otherwise it's an image grid.
        bins (bool, optional): Number of bins for histogram, if `histogram` is
            True (default 100).
    
    Note:
        * If modules or match_names are not provided then no hooks will be
          attached.
    """
    hook = _hook_generator(do_grad_input,do_grad_output,tag,save_path,replace, histogram, bins, 'backward')    
    _register(net, hook, modules, match_names, False)
    return net

def parameters_hook(net, modules=None, match_names=None, param_names=None,
                    tag='', save_path='.', replace=True, histogram=True, bins=100):
    """Registers a forward hook to a network's modules for vizualization of its parameters.

    When net.forward() is called, the hook saves an image grid or a histogram 
    of the parameters of the specified modules.

    Args:
        net (nn.Module): The network whose parameters are to be visualized.
        modules (list or tuple, optional): List of class definitions for the
            modules where the hook is attached e.g. nn.Conv2d  (default None).
        match_names (list or tuple, optional): List of strings. If any modules
            contain one of the strings then the hook is attached (default None).
        param_names (list or tuple, optional): List of strings. If any
            parameters of the module contain one of the strings then they are
            visualized (default None).
        tag (str, optional): String tag to attach to saved images (default None).
        save_path (str, optional): Path to save visualisation results 
            (default '.').
        replace (bool, optional): If True, the images (from the same module) 
            are replaced whenever the hook is called (default True).
        histogram (bool, optional): If True then the visualization is a
            histrogram, otherwise it's an image grid.
        bins (bool, optional): Number of bins for histogram, if `histogram` is
            True (default 100).

    Note:
        * If modules or match_names are not provided then no hooks will be
          attached.
        * If param_names are not provided then no parameters will be visualized.
    """
    hook = _hook_generator(False,False,tag,save_path,replace, histogram, bins, 'parameters', param_names)    
    _register(net, hook, modules, match_names, True)
    return net

def parameters(net, modules=None, match_names=None, param_names=None, tag='', save_path='.', histogram=True, bins=100):
    """Visualizes a network's parameters on an image grid or histogram.

    Args:
        net (nn.Module): The network whose parameters are to be visualized.
        modules (list or tuple, optional): List of class definitions for the
            modules where the hook is attached e.g. nn.Conv2d  (default None).
        match_names (list or tuple, optional): List of strings. If any modules
            contain one of the strings then the hook is attached (default None).
        param_names (list or tuple, optional): List of strings. If any
            parameters of the module contain one of the strings then they are
            visualized (default None).
        tag (str, optional): String tag to attach to saved images (default None).
        save_path (str, optional): Path to save visualisation results 
            (default '.').
        histogram (bool, optional): If True then the visualization is a
            histrogram, otherwise it's an image grid.
        bins (bool, optional): Number of bins for histogram, if `histogram` is
            True (default 100).

    Note:
        * If modules or match_names are not provided then no parameters will be
          visualized.
        * If param_names are not provided then no parameters will be visualized.
    """
    save_path = process(save_path, True)
    modules = process_none(modules)
    match_names = process_none(match_names)
    for module_name, mod in net.named_modules():
        name_match = any([torch.typename(modules).find(x) >= 0 for x in match_names])
        instance_match = any([isinstance(mod, x) for x in modules])
        if instance_match or name_match:
            params = {x: _get_tensor(getattr(mod, x)) for x in param_names}
            for tensor_name, data in params.items():
                title = '{0}-{1}-{2}'.format(tag, module_name, tensor_name)
                if data is None:
                    continue
                if histogram:
                    img = torch2cv(data)
                    df = pd.DataFrame(img.reshape(img.size))
                    fig, ax = plt.subplots()
                    df.hist(bins=bins, ax=ax)
                    fig.savefig(os.path.join(save_path, '{0}.png'.format(title)))
                    plt.close(fig)
                else:
                    if data.dim() > 1:
                        img = torch2cv(make_grid(data, color=False))
                        to_save = (map_range(img)*255).astype(int)
                        cv2.imwrite(os.path.join(save_path, '{0}.png'.format(title)), to_save)