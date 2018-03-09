import math
import torch
from torch import nn
import numpy as np


def count_parameters(net):
    """Counts the parameters of a given PyTorch model."""
    return sum([p.numel() for p in list(net.parameters())])

def str2bool(x):
    """Converts a string to boolean type.
    
    If the string is any of ['no', 'false', 'f', '0'], or any capitalization,
    e.g. 'fAlSe' then returns False. All other strings are True.

    """
    if x is None or x.lower() in ['no', 'false', 'f', '0']:
        return False
    else:
        return True
    
def str_is_int(s):
    """Checks if a string can be converted to int."""
    try:
        int(s)
        return True
    except:
        return False

def slide_window_(a, kernel, stride=None):
    """Expands last dimension to help compute sliding windows.
    
    Args:
        a (Tensor or Array): The Tensor or Array to view as a sliding window.
        kernel (int): The size of the sliding window.
        stride (tuple or int, optional): Strides for viewing the expanded dimension (default 1)

    The new dimension is added at the end of the Tensor or Array.

    Returns:
        The expanded Tensor or Array.

    Running Sum Example::

        >>> a = torch.Tensor([1, 2, 3, 4, 5, 6])
         1
         2
         3
         4
         5
         6
        [torch.FloatTensor of size 6]
        >>> a_slided = dlt.util.slide_window_(a.clone(), kernel=3, stride=1)
         1  2  3
         2  3  4
         3  4  5
         4  5  6
        [torch.FloatTensor of size 4x3]
        >>> running_total = (a_slided*torch.Tensor([1,1,1])).sum(-1)
          6
          9
         12
         15
        [torch.FloatTensor of size 4]

    Averaging Example::

        >>> a = torch.Tensor([1, 2, 3, 4, 5, 6])
         1
         2
         3
         4
         5
         6
        [torch.FloatTensor of size 6]
        >>> a_sub_slide = dlt.util.slide_window_(a.clone(), kernel=3, stride=3)
         1  2  3
         4  5  6
        [torch.FloatTensor of size 2x3]
        >>> a_sub_avg = (a_sub_slide*torch.Tensor([1,1,1])).sum(-1) / 3.0
         2
         5
        [torch.FloatTensor of size 2]
    """


    if isinstance(kernel, int):
        kernel = (kernel,)
    if stride is None:
        stride = tuple(1 for i in kernel)
    elif isinstance(stride, int):
        stride = (stride,)
    window_dim = len(kernel)
    if is_array(a):
        new_shape = a.shape[:-window_dim] + tuple(int(np.floor((s - kernel[i] )/stride[i]) + 1) for i,s in enumerate(a.shape[-window_dim:])) + kernel
        new_stride = a.strides[:-window_dim] + tuple(s*k for s,k in zip(a.strides[-window_dim:], stride)) + a.strides[-window_dim:]
        return np.lib.stride_tricks.as_strided(a, shape=new_shape, strides=new_stride)
    else:
        new_shape = a.size()[:-window_dim] + tuple(int(np.floor((s - kernel[i] )/stride[i]) + 1) for i,s in enumerate(a.size()[-window_dim:])) + kernel
        new_stride = a.stride()[:-window_dim] + tuple(s*k for s,k in zip(a.stride()[-window_dim:], stride)) + a.stride()[-window_dim:]
        a.set_(a.storage(), storage_offset=0, size=new_shape, stride=new_stride)
        return a

# TODO: Add example
def re_stride(a, kernel, stride=None):
    """Returns a re-shaped and re-strided tensor/variable given a kernel (uses as_strided).

    Args:
        a (Tensor): The Tensor to re-stride.
        kernel (tuple or int): The size of the new dimension(s).
        stride (tuple or int, optional): Strides for viewing the expanded dimension(s) (default 1)
    """
    if isinstance(kernel, int):
        kernel = (kernel,)
    if stride is None:
        stride = tuple(1 for i in kernel)
    elif isinstance(stride, int):
        stride = (stride,)
    window_dim = len(kernel)
    new_shape = a.size()[:-window_dim]  + kernel + tuple(int(math.floor((s - kernel[i] )/stride[i]) + 1) for i,s in enumerate(a.size()[-window_dim:]))
    new_stride = a.stride()[:-window_dim]  + a.stride()[-window_dim:] + tuple(s*k for s,k in zip(a.stride()[-window_dim:], stride))
    return a.as_strided(new_shape, new_stride)


def replicate(x, dim=-3, nrep=3):
    """Replicates Tensor/Array in a new dimension.

    Args:
        x (Tensor or Array): Tensor to replicate.
        dim (int, optional): New dimension where replication happens.
        nrep (int, optional): Number of replications.
    """
    if is_tensor(x) or is_variable(x):
        return x.unsqueeze(dim).expand(*x.size()[:dim + 1],nrep,*x.size()[dim + 1:])
    else:
        return np.repeat(np.expand_dims(x,dim), nrep, axis=dim)

def moving_avg(x, width=5):
    """Performes moving average of a one dimensional Tensor or Array

    Args:
        x (Tensor or Array): 1D Tensor or array.
        width (int, optional): Width of the kernel.
    """
    if len(x) >= width:
        if is_array(x):
            return np.mean(slide_window_(x, width,1), -1)
        else:
            return torch.mean(slide_window_(x, width,1), -1)
    else:
        return x.mean()

def moving_var(x, width=5):
    """Performes moving variance of a one dimensional Tensor or Array

    Args:
        x (Tensor or Array): 1D Tensor or array.
        width (int, optional): Width of the kernel.
    """
    if len(x) >= width:
        if is_array(x):
            return np.var(slide_window_(x, width, 1), -1)
        else:
            return torch.var(slide_window_(x, width, 1), -1)
    else:
        return x.var()

def sub_avg(x, width=5):
    """Performes averaging of a one dimensional Tensor or Array every `width` elements.

    Args:
        x (Tensor or Array): 1D Tensor or array.
        width (int, optional): Width of the kernel.
    """
    if len(x) >= width:
        if is_array(x):
            return np.mean(slide_window_(x, width, width), -1)
        else:
            return torch.mean(slide_window_(x, width, width), -1)
    else:
        return x.mean()

def sub_var(x, width=5):
    """Calculates variance of a one dimensional Tensor or Array every `width` elements.

    Args:
        x (Tensor or Array): 1D Tensor or array.
        width (int, optional): Width of the kernel.
    """
    if len(x) >= width:
        if is_array(x):
            return np.var(slide_window_(x, width, width), -1)
        else:
            return torch.var(slide_window_(x, width, width), -1)
    else:
        return x.var()

def has(x, val):
    """Checks if a Tensor/Array has a value val. Does not work with nan (use :func:`has_nan`)."""
    return bool((x == val).sum() != 0)

def has_nan(x):
    """Checks if a Tensor/Array has NaNs."""
    return bool((x != x).sum() > 0)


def has_inf(x):
    """Checks if a Tensor/Array array has Infs."""
    return has(x, float('inf'))

def replace_specials_(x, val=0):
    """Replaces NaNs and Infs from a Tensor/Array.
    
    Args:
        x (Tensor or Array): The Tensor/Array (gets replaced in place).
        val (int, optional): Value to replace NaNs and Infs with (default 0).
    """
    x[ (x == float('inf')) | (x != x) ] = val
    return x

def replace_inf_(x, val=0):
    """Replaces Infs from a Numpy Array.
    
    Args:
        x (Array): The Array (gets replaced in place).
        val (int, optional): Value to replace Infs with (default 0).
    """
    x[x == float('inf')] = val
    return x

def replace_nan_(x, val=0):
    """Replaces NaNs from a Numpy Array.

    Args:
        x (Array): The Array (gets replaced in place).
        val (int, optional): Value to replace Infs with (default 0).
    """
    x[x != x] = val
    return x

def map_range(x, low=0, high=1):
    """Maps the range of a Numpy Array to [low, high] globally."""
    if is_array(x):
        return np.interp(x, [x.min(), x.max()], [low, high]).astype(x.dtype)
    else:
        xmax, xmin = x.max(), x.min()
        if xmax - xmin == 0:
            return x.clone().fill_(0)
        return x.add(-xmin).div_(xmax-xmin).mul_(high-low).add_(low).clamp_(low, high)

def is_variable(x):
    """Checks if input is a Variable instance."""
    return isinstance(x, torch.autograd.Variable)

# This was added to torch in v0.3. Keeping it here too.
def is_tensor(x):
    """Checks if input is a Tensor"""
    return torch.is_tensor(x)
    
def is_cuda(x):
    """Checks if input is a cuda Tensor."""
    return torch.is_tensor(x) and x.is_cuda

def is_array(x):
    """Checks if input type contains 'array' or 'series' in its typename."""
    return torch.typename(x).find('array') >= 0 or torch.typename(x).find('series') >= 0 

## Returns a numpy array version of x
def to_array(x):
    """Converts x to a Numpy Array.
    
    Args:
        x (Variable, Tensor or Array): Input to be converted. Can also be on the GPU.

    Automatically gets the data from torch Variables and casts GPU Tensors
    to CPU.
    """
    if is_variable(x):
        x = x.data.clone()
    if is_cuda(x):
        x = x.cpu()
    if is_tensor(x):
        return x.numpy()
    else:
        return x.copy()

## Returns a cpu tensor COPY version of x
def to_tensor(x):
    """Converts x to a Torch Tensor (CPU).
    
    Args:
        x (Variable, Tensor or Array): Input to be converted. Can also be on the GPU.

    Automatically gets the data from torch Variables and casts GPU Tensors to cpu.
    to CPU.
    """
    if is_variable(x):
        return x.data.clone()
    if is_cuda(x):
        return x.cpu()
    if is_array(x):
        return torch.from_numpy(x)
    else:
        return x.clone()

########
### Variables, tensors, arrays, cuda, cpu, image views etc
########
def permute(x, perm):
    """Permutes the last three dimensions of the input Tensor or Array.

    Args:
        x (Tensor or Array): Input to be permuted.
        perm (tuple or list): Permutation.

    Note:
        If the input has less than three dimensions a copy is returned.
    """
    if is_tensor(x):
        if x.dim() < 3:
            return x.clone()
        else:     
            s = tuple(range(0, x.dim()))
            permutation = s[:-3] + tuple(s[-3:][i] for i in perm)
        return x.permute(*permutation).contiguous()
    elif is_array(x):
        if x.ndim < 3:
            return x.copy()
        else:
            s = tuple(range(0,x.ndim))
            permutation = s[:-3] + tuple(s[-3:][i] for i in perm)
        # Copying to get rid of negative strides
        return np.transpose(x, permutation).copy()
    else:
        raise TypeError('Uknown type {0} encountered.'.format(torch.typename(x)))

def hwc2chw(x):
    """Permutes the last three dimensions of the hwc input to become chw.

    Args:
        x (Tensor or Array): Input to be permuted.
    """
    return permute(x, (2,0,1))
def chw2hwc(x):
    """Permutes the last three dimensions of the chw input to become hwc.

    Args:
        x (Tensor or Array): Input to be permuted.
    """
    return permute(x, (1,2,0))

def channel_flip(x, dim=-3):
    """Reverses the channel dimension.
    
    Args:
        x (Tensor or Array): Input to have its channels flipped.
        dim (int, optional): Channels dimension (default -3).

    Note:
        If the input has less than three dimensions a copy is returned.
    """

    if is_tensor(x) or is_variable(x):
        dim = x.dim() + dim if dim < 0 else dim
        if x.dim() < 3:
            return x.clone()
        return x[tuple(slice(None, None) if i != dim
                 else torch.arange(x.size(i)-1, -1, -1).long()
                 for i in range(x.dim()))]
    elif is_array(x):
        dim = x.ndim + dim if dim < 0 else dim
        if x.ndim < 3:
            return x.copy()
        return np.ascontiguousarray(np.flip(x,dim))
    else:
        raise TypeError('Uknown type {0} encountered.'.format(torch.typename(x)))

# Default is dimension -3 (e.g. for bchw)
def rgb2bgr(x, dim=-3):
    """Reverses the channel dimension. See :func:`channel_flip`"""
    return channel_flip(x, dim)
    
def bgr2rgb(x, dim=-3):
    """Reverses the channel dimension. See :func:`channel_flip`"""
    return channel_flip(x, dim)

# Getting images from one library to the other
# Always assuming the last three dimensions are the images
# opencv is hwc - BGR
# torch is chw - RGB
# plt is hwc - RGB
VIEW_NAMES = {
    'opencv': ['hwcbgr', 'hwc-bgr', 'bgrhwc', 'bgr-hwc', 'opencv', 'open-cv', 'cv', 'cv2'],
    'torch':  ['chwrgb', 'chw-rgb', 'rgbchw', 'rgb-chw', 'torch', 'pytorch'],
    'plt':    ['hwcrgb', 'hwc-rgb', 'rgbhwc', 'rgb-hwc', 'plt', 'pyplot', 'matplotlib'],
    'other':  ['chwbgr', 'chw-bgr', 'bgrchw', 'bgr-chw']
}

def _determine_view(v):
    for view, names in VIEW_NAMES.items():
        if v.lower() in names:
            return view
    return 'unknown'

# This is not elegant but at least it's clear and does its job
def change_view(x, current, new):
    """Changes the view of the input. Returns a copy.

    Args:
        x (Tensor or Array): Input whose view is to be changed.
        current (str): Current view.
        new (str): New view.

    Possible views:

    ======== ==============================================================
      View     Aliases
    ======== ==============================================================
     opencv   hwcbgr, hwc-bgr, bgrhwc, bgr-hwc, opencv, open-cv, cv, cv2
    -------- --------------------------------------------------------------
     torch    chwrgb, chw-rgb, rgbchw, rgb-chw, torch, pytorch
    -------- --------------------------------------------------------------
     plt      hwcrgb, hwc-rgb, rgbhwc, rgb-hwc, plt, pyplot, matplotlib
    -------- --------------------------------------------------------------
     other    chwbgr, chw-bgr, bgrchw, bgr-chw
    ======== ==============================================================

    Note:
        If the input has less than three dimensions a copy is returned.    

    """
    curr_name, new_name = _determine_view(current), _determine_view(new)
    if curr_name == 'unknown':
        raise ValueError('Unkown current view encountered in change_view: {0}'.format(current))
    if new_name == 'unknown':
        raise ValueError('Unkown new view encountered in change_view: {0}'.format(new))
    if new_name == curr_name:
        if is_array(x):
            return x.copy()
        else:
            return x.clone()

    if curr_name == 'opencv':
        if new_name == 'torch':
            return bgr2rgb(hwc2chw(x), -3)
        elif new_name == 'plt':
            return bgr2rgb(x, -1)
        elif new_name == 'other':
            return hwc2chw(x)
    if curr_name == 'torch':
        if new_name == 'opencv':
            return chw2hwc(rgb2bgr(x, -3))
        elif new_name == 'plt':
            return chw2hwc(x)
        elif new_name == 'other':
            return rgb2bgr(x, -3)
    if curr_name == 'plt':
        if new_name == 'torch':
            return hwc2chw(x)
        elif new_name == 'opencv':
            return rgb2bgr(x, -1)
        elif new_name == 'other':
            return hwc2chw(rgb2bgr(x, -1))
    if curr_name == 'other':
        if new_name == 'torch':
            return bgr2rgb(x, -3)
        elif new_name == 'plt':
            return chw2hwc(rgb2bgr(x, -3))
        elif new_name == 'opencv':
            return chw2hwc(x)

## These functions also convert!
def cv2torch(x):
    """Converts input to Tensor and changes view from cv (hwc-bgr) to torch (chw-rgb).
    
    For more detail see :func:`change_view`
    """
    return change_view(to_tensor(x), 'cv', 'torch')

def torch2cv(x):
    """Converts input to Array and changes view from torch (chw-rgb) to cv (hwc-bgr).

    For more detail see :func:`change_view`
    """
    return change_view(to_array(x), 'torch', 'cv')

def cv2plt(x):
    """Changes view from cv (hwc-bgr) to plt (hwc-rgb).
        
    For more detail see :func:`change_view`
    """
    return change_view(x, 'cv', 'plt')

def plt2cv(x):
    """Changes view from plt (hwc-rgb) to cv (hwc-bgr).
        
    For more detail see :func:`change_view`
    """
    return change_view(x, 'plt', 'cv')

def plt2torch(x):
    """Converts input to Tensor and changes view from plt (hwc-rgb) to torch (chw-rgb).
    
    For more detail see :func:`change_view`
    """
    return change_view(to_tensor(x), 'plt', 'torch')

def torch2plt(x):
    """Converts input to Array and changes view from torch (chw-rgb) to plt (hwc-rgb) .
    
    For more detail see :func:`change_view`
    """
    return change_view(to_array(x), 'torch', 'plt')


# Functions for backward/forward compatibility..!
def _get_torch_version():
    class VRS(object):
        def __init__(self, major, minor):
            self.major = major
            self.minor = minor
    try:
        vrs_split = torch.__version__.split('.')
        major = vrs_split[0]
        minor = vrs_split[1]
    except:
        print('Unknown torch version')
        return VRS(-1, -1)
    if major[0] == 'v':
        major = int(major[1:])
    else:
        major = int(major)
    minor = int(minor)
    return VRS(major, minor)

_torch_version = _get_torch_version()

if _torch_version.minor > 3 and _torch_version.major == 0:
    _get_scalar_value = lambda x: x.item()
else:
    _get_scalar_value = lambda x: x[0]