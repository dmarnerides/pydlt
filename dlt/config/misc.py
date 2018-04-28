import os
from ..hdr import imwrite
from ..util import to_array, make_grid, change_view, applier
from ..util.paths import process
from ..viz import imshow
from .opts import fetch_opts

# Saving samples

def sample_images(samples, tag='', view='torch', ext='.png', color=True,
                  size=None, inter_pad=None, fill_value=0, preprocess=None,
                  subset=None):
    """Saves image samples. (Partly) configured using command line arguments.

    Args:
        samples (Tensor, Array, list or tuple): Image samples to save. Will
            automatically be put in a grid. Must be in the [0,1] range.
        tag (str, optional): Tag to add to the saved samples, e.g. epoch
            (default None).
        view (str, optional): The image view e.g. 'hwc-bgr' or 'torch'
            (default 'torch').
        ext (str, optional): The image format for the saved samples (default '.png').
        color (bool, optional): Treat images as colored or not (default True).
        size (list or tuple, optional): Grid dimensions, rows x columns. (default None).
        inter_pad (int, optional): Padding separating the images (default None).
        fill_value (int, optional): Fill value for inter-padding (default 0).
        preprocess (callable, optional): Pre processing to apply to each image 
            sample (default None).
        subset (string, optional): Specifies the subset of the relevant
            categories, if any of them was split (default, None).

    Relevant Command Line Arguments:
        - **general**: `--save_path`.
        - **samples**: `--display_samples`, `--save_samples`, `--sample_freq`.

    Note:
        Settings are automatically acquired from a call to :func:`dlt.config.parse`
        from the built-in ones. If :func:`dlt.config.parse` was not called in the 
        main script, this function will call it.
    """

    opts = fetch_opts(['general', 'samples'], subset)
    def get_sample(x):
        if preprocess:
            x = preprocess(x)
        return to_array(x)
    if (opts.save_samples or opts.display_samples) \
            and (sample_images.count % opts.sample_freq == 0):
        name = subset['samples'] + '_' if isinstance(subset, dict) else subset + '_' if subset is not None else ''
        tag = tag + '_' if tag != '' else tag
        imname = '{0}{1}{2:07d}{3}'.format(name, tag, sample_images.count + 1, ext)
        
        grid = make_grid(applier(get_sample)(samples),
                         view=view, color=color, size=size,
                         inter_pad=inter_pad, fill_value=fill_value)
        grid = change_view(grid, view, 'cv')

        if opts.display_samples:
            figure = sample_images.figures.get(name)
            sample_images.figures['name'] = imshow(grid, interactive=True,
                                                title=imname, figure=figure, 
                                                view='cv')

        if opts.save_samples:
            if ext not in ['.hdr', '.pfm', '.exr']:
                grid = (grid*255).astype(int)
            sample_path = process(os.path.join(opts.save_path, 'samples'), True)
            imname = os.path.join(sample_path, imname)
            imwrite(imname, grid)

    sample_images.count += 1

sample_images.count = 0
sample_images.figures = {}