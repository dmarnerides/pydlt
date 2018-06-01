import os
from ..hdr import imwrite
from ..util import to_array, make_grid, change_view, applier
from ..util.paths import process
from ..viz import imshow
from .opts import fetch_opts


class ImageSampler(object):
    """Saves and/or displays images. (Partly) configured using command line arguments.

    Args:
        view (str, optional): The image view e.g. 'hwc-bgr' or 'torch'
            (default 'torch').
        ext (str, optional): The image format for the saved samples (default '.jpg').
        color (bool, optional): Treat images as colored or not (default True).
        size (list or tuple, optional): Grid dimensions, rows x columns. (default None).
        inter_pad (int, optional): Padding separating the images (default None).
        fill_value (int, optional): Fill value for inter-padding (default 0).
        preprocess (callable, optional): Pre processing to apply to the image 
            samples (default None).
        subset (string, optional): Specifies the subset of the relevant
            categories, if any of them were split (default, None).

    Relevant Command Line Arguments:
        - **general**: `--save_path`.
        - **samples**: `--display_samples`, `--save_samples`, `--sample_freq`.

    Note:
        Settings are automatically acquired from a call to :func:`dlt.config.parse`
        from the built-in ones. If :func:`dlt.config.parse` was not called in the 
        main script, this function will call it.
    """
    def __init__(self, view='torch', ext='.jpg', color=True, size=None,
                 inter_pad=None, fill_value=0, preprocess=None, subset=None):
        self.view = view
        self.ext = ext
        self.preprocess = preprocess
        self.color = color
        self.size = size
        self.inter_pad = inter_pad
        self.fill_value = fill_value
        self.count = 0
        self.figure = None
        if subset is None:
            self.name = ''
        elif isinstance(subset, dict):
            self.name = subset['samples']
        else:
            self.name = subset + '_'
        self.opts = fetch_opts(['general', 'samples'], subset)
        self.sample_path = process(os.path.join(self.opts.save_path, 'samples'), True)
    
    def sample(self, imgs, tag=''):
        """Saves and/or displays functions depending on the configuration.
        
        Args:
            imgs (Tensor, Array, list or tuple): Image samples. Will
                automatically be put in a grid. Must be in the [0,1] range.
            tag (str, optional): Tag to add to the saved samples, e.g. epoch
                (default None).
        """
        if (self.opts.save_samples or self.opts.display_samples) \
            and (self.count % self.opts.sample_freq == 0):
            samples = self._get_sample(imgs)
            if self.opts.save_samples:
                self._save(samples, tag)
            if self.opts.display_samples:
                self._display(samples, tag)
        self.count += 1

    def __call__(self, imgs, tag=''):
        """Same as :meth:`sample`"""
        self.sample(imgs)

    def _display(self, img, tag=''):
        self.figure = imshow(img, interactive=True, title=self._make_name(tag), 
                             figure=self.figure, view='cv')

    def _save(self, img, tag=''):
        full_name = os.path.join(self.sample_path, self._make_name(tag))
        print(full_name)
        imwrite(full_name, img)
        

    def _make_name(self, tag):
        tag = tag + '_' if tag != '' else tag
        tmpl = '{0}{1}{2:07d}{3}'
        return tmpl.format(self.name, tag, self.count + 1, self.ext)

    def _get_sample(self, img):
        if self.preprocess:
            img = self.preprocess(img)
        img = applier(to_array)(img)
        grid = make_grid(img, view=self.view, color=self.color, size=self.size,
                         inter_pad=self.inter_pad, fill_value=self.fill_value)
        grid = change_view(grid, self.view, 'cv')
        if self.ext not in ['.hdr', '.pfm', '.exr']:
            grid = (grid*255).astype('uint8')
        return grid

    def __getstate__(self):
        return self.state_dict()

    def __setstate__(self, state):
        self.load_state_dict(state)

    def state_dict(self):
        return {key: value for key, value in self.__dict__.items()
                if key not in ['preprocess', 'figure']}

    def load_state_dict(self, state_dict):
        self.__dict__.update(state_dict)