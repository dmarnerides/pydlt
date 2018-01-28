import math

def _find_integers(down, up, include_down, include_up):

    ret = []
    if up < 0 or up < down:
        return ret
    if include_down and down >= 0 and float(down).is_integer():
        ret.append(int(down))
    for i in range(int(math.floor(down + 1)), int(math.ceil(up - 1)) + 1):
        if i >= 0:
            ret.append(i)
    if include_up and float(up).is_integer():
        ret.append(int(up))
    return ret

def out_size(dim_in, k, s, p, d):
    """Calculates the resulting size after a convolutional layer.
    
    Args:
        dim_in (int): Input dimension size.
        k (int): Kernel size.
        s (int): Stride of convolution.
        p (int): Padding (of input).
        d (int): Dilation
    """
    return math.floor((dim_in + 2*p - d*(k-1) - 1)/s + 1)

def in_size(dim_out, k, s, p, d):
    """Calculates the input size before a convolutional layer.
    
    Args:
        dim_out (int): Output dimension size.
        k (int): Kernel size.
        s (int): Stride of convolution.
        p (int): Padding (of input).
        d (int): Dilation
    """
    down = s*(dim_out - 1) + d*(k - 1) + 1 - 2*p
    up = s*dim_out + d*(k - 1) + 1 - 2*p
    return _find_integers(down, up, True, False)

def kernel_size(dim_in, dim_out, s, p, d):
    """Calculates the possible kernel size(s) of a convolutional layer given input and output.
    
    Args:
        dim_in (int): Input dimension size.
        dim_out (int): Output dimension size.
        s (int): Stride of convolution.
        p (int): Padding (of input).
        d (int): Dilation
    """
    down = ((dim_in + 2*p - s*dim_out - 1) / d) + 1
    up = ((dim_in + 2*p - s*(dim_out - 1) -1) / d) + 1
    return _find_integers(down, up, False, True)

def stride_size(dim_in, dim_out, k, p, d):
    """Calculates the possible stride size(s) of a convolutional layer given input and output.
    
    Args:
        dim_in (int): Input dimension size.
        dim_out (int): Output dimension size.
        k (int): Kernel size.
        p (int): Padding (of input).
        d (int): Dilation
    """
    down = (dim_in + 2*p -d*(k-1) - 1) / dim_out
    up = (dim_in + 2*p -d*(k-1) - 1) / (dim_out -1)
    return _find_integers(down, up, False, True)

def padding_size(dim_in, dim_out, k, s, d):
    """Calculates the possible padding size(s) of a convolutional layer given input and output.
    
    Args:
        dim_in (int): Input dimension size.
        dim_out (int): Output dimension size.
        k (int): Kernel size.
        s (int): Stride of convolution.
        d (int): Dilation
    """
    down = (s*(dim_out - 1) + d*(k-1) - dim_in + 1)/2
    up = (s*dim_out + d*(k-1) - dim_in + 1)/2
    return _find_integers(down, up, True, False)

def dilation_size(dim_in, dim_out, k, s, p):
    """Calculates the possible dilation size(s) of a convolutional layer given input and output.

    Args:
        dim_in (int): Input dimension size.
        dim_out (int): Output dimension size.
        k (int): Kernel size.
        s (int): Stride of convolution.
        p (int): Padding (of input).
    """
    if k <= 1:
        print('Warning: Invalid kernel size {0} for dilation size prediction.'.format(k))
        return []
    down = (dim_in + 2*p - s*dim_out - 1)/(k-1)
    up = (dim_in + 2*p - s*(dim_out - 1) - 1)/(k-1)
    return _find_integers(down, up, False, True)

def _make_list(x):
    if isinstance(x,int):
        return [x]
    else:
        return x

def find_layers(dims_in=None, dims_out=None, ks=None, ss=None, ps=None, ds=None):
    """Calculates all the possible convolutional layer size(s) and parameters.

    Args:
        dim_in (list): Input dimension sizes.
        dim_out (list): Output dimension sizes.
        k (list): Kernel sizes.
        s (list): Strides of convolutions.
        p (list): Paddings (of inputs).
    """
    if isinstance(dims_in,int):
        dims_in = [dims_in]
    in_dims = range(16, 33) if dims_in is None else _make_list(dims_in)
    out_dims = range(16, 33) if dims_out is None else _make_list(dims_out)
    kernels = range(1, 5) if ks is None else _make_list(ks)
    strides = range(1, 3) if ss is None else _make_list(ss)
    paddings = range(1, 33) if ps is None else _make_list(ps)
    dilations = range(1, 9) if ds is None else _make_list(ds)
    result = []
    for kernel in kernels:
        for stride in strides:
            for padding in paddings:
                for dilation in dilations:
                    for out_dim in out_dims:
                        in_dim_preds = in_size(out_dim, kernel, stride, padding, dilation)
                        for in_dim_pred in in_dim_preds:
                            if in_dim_pred in in_dims:
                                result.append([in_dim_pred, out_dim, kernel, stride, padding, dilation])

    return result


