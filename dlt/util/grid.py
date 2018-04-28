import torch
import numpy as np
from .misc import _determine_view, change_view, is_array, is_tensor, \
                  to_array, to_tensor, map_range

def make_grid(images, view='torch', color=True, size=None, inter_pad=None, fill_value=0, scale_each=False):
    """Creates a single image grid from a set of images.

    Args:
        images (Tensor, Array, list or tuple): Torch Tensor(s) and/or Numpy Array(s). 
        view (str, optional): The image view e.g. 'hwc-bgr' or 'torch'
            (default 'torch').
        color (bool, optional): Treat images as colored or not (default True).
        size (list or tuple, optional): Grid dimensions, rows x columns. (default None).
        inter_pad (int or list/tuple, optional): Padding separating the images (default None).
        fill_value (int, optional): Fill value for inter-padding (default 0).
        scale_each (bool, optional): Scale each image to [0-1] (default False).

    Returns:
        Tensor or Array: The resulting grid. If any of the inputs is an Array
        then the result is an Array, otherwise a Tensor.

    Notes:
        - Images of **different sizes are padded** to match the largest.
        - Works for **color** (3 channels) or **grey** (1 channel/0 channel)
          images.
        - Images must have the same view (e.g. chw-rgb (torch))
        - The Tensors/Arrays can be of **any dimension >= 2**. The last 2 (grey)
          or last 3 (color) dimensions are the images and all other dimensions
          are stacked. E.g. a 4x5x3x256x256 (torch view) input will be treated:

            - As 20 3x256x256 color images if color is True.
            - As 60 256x256 grey images if color is False.
        
        - If color is False, then only the last two channels are considered
          (as hw) thus any colored images will be split into their channels.
        - The image list can contain both **Torch Tensors and Numpy Arrays**.
          at the same time as long as they have the same view.
        - If size is not given, the resulting grid will be the smallest square
          in which all the images fit. If the images are more than the given
          size then the default smallest square is used.

    Raises:
        TypeError: If images are not Arrays, Tensors, a list or a tuple
        ValueError: If channels or dimensions are wrong.

    """


    # Determine view
    orig_view = _determine_view(view)
    if orig_view == 'unknown':
        print('make_grid provided with unknown view: ' + view)
        
    # Flag if we need to convert back to array
    should_convert_to_array = False
    
    if torch.typename(images) in ['tuple', 'list']:
        # Handle tuple and list
        # First convert to tensor
        should_convert_to_array = any([is_array(x) for x in images])
        images = [to_tensor(im) for im in images]
        # Change view only if mode is color (otherwise last 2 dimensions are hw)
        if color:
            # Change view to torch
            if orig_view != 'torch':
                images = [change_view(x, orig_view, 'torch') for x in images]
            # Must have more than 2 dimensions
            if any([x.dim() <= 2 for x in images]):
                raise ValueError('A provided image has less than three dimensions. '
                                 'Maybe you wanted to pass color=False?')
            # Must have 3 channels
            if any([x.size(-3) != 3 for x in images]):
                raise ValueError('A provided image does not have 3 channels. '
                                 'Maybe you wanted to pass color=False or a different view?')
            # Make all tensors 4d
            images = [x.unsqueeze(0) if x.dim() == 3 else x.view(-1, *x.size()[-3:]) for x in images]
            
        else:
            # Make all tensors 4d with
            images = [x.unsqueeze(0).unsqueeze(0).view(-1, 1, *x.size()[-2:]) for x in images]
        # Pad images to match largest
        maxh, maxw = max([x.size(-2) for x in images]), max([x.size(-1) for x in images])
        for i, img in enumerate(images):
            imgh, imgw = img.size(-2), img.size(-1)
            if (img.size(-2) < maxh) or (img.size(-1) < maxw):
                padhl = int((maxh - imgh)/2)
                padhr = maxh - imgh - padhl
                padwl = int((maxw - imgw)/2)
                padwr = maxw - imgw - padwl
                images[i] = torch.nn.functional.pad(img, (padwl, padwr, padhl, padhr))
        images = torch.cat(images,0)
    elif is_tensor(images) or is_array(images):
        should_convert_to_array = is_array(images)
        images = to_tensor(images)
        images = change_view(images, orig_view, 'torch')
        if color:
            if images.size()[-3] != 3:
                raise ValueError('A provided image does not have 3 channels. '
                                 'Maybe you wanted to pass color=False or a different view?')
            images = images.unsqueeze(0).view(-1,3,*images.size()[-2:])
        else:
            images = images.unsqueeze(0).unsqueeze(0).view(-1,1,*images.size()[-2:])
    else:
        raise TypeError('make_grid can only accept tuples, lists, tensors'
                         ' and numpy arrays. Got {0}'.format(torch.typename(images)))

    # Scale each
    if scale_each:
        for i in range(images.size(0)):
            images[i] = map_range(images[i])

    ### Finally create grid
    n_images, n_chan, im_w, im_h = images.size()[0], images.size()[1],images.size()[2], images.size()[3]    
    # Get number of columns and rows (width and height)
    if (size is not None and n_images > size[0] * size[1]) or size is None:
        n_row = int(np.ceil(np.sqrt(n_images)))
        n_col = int(np.ceil(n_images / n_row))
    else:
        n_col = size[0]
        n_row = size[1]
    
    if inter_pad is not None:
        if isinstance(inter_pad, int):
            inter_pad = (inter_pad, inter_pad)
        w_pad, h_pad = inter_pad[1], inter_pad[0]
        total_w_padding,  total_h_padding = max(w_pad,0) * (n_col - 1), max(h_pad,0) * (n_row - 1)
    else:
        w_pad, h_pad = 0, 0
        total_w_padding,  total_h_padding = 0, 0

    w,h = int(im_w * n_col) + total_w_padding, int(im_h * n_row) + total_h_padding
    grid = torch.Tensor(n_chan,w,h).type_as(images).fill_(fill_value)
    for i in range(n_images):
        i_row = i % n_row
        i_col = int(i/n_row)
        grid[:,
             i_col*(im_w + w_pad):(i_col)*(im_w + w_pad) + im_w,
             i_row*(im_h + h_pad):(i_row)*(im_h + h_pad) + im_h].copy_(images[i])

    if should_convert_to_array:
        grid = to_array(grid)
    if orig_view != 'torch':
        grid = change_view(grid, 'torch', orig_view)
    return grid
