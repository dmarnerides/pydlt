import os
from os import path
import torch
from ..hdr import imwrite
from ..util import to_array, make_grid, change_view, applier
from ..util.paths import process
from ..viz import imshow


class ImageSampler(object):
    """Saves and/or displays image samples.

    Args:
        name (str): Name of the checkpointer. This will also be used for
                the checkpoint filename.
        directory (str, optional): Parent directory of where the samples will be saved.
            A new sub-directory called samples will be created (default '.').
        overwrite (bool, optional): Overwrite/remove the previous checkpoint (default False).
        view (str, optional): The image view e.g. 'hwc-bgr' or 'torch'
            (default 'torch').
        ext (str, optional): The image format for the saved samples (default '.jpg').
        color (bool, optional): Treat images as colored or not (default True).
        size (list or tuple, optional): Grid dimensions, rows x columns. (default None).
        inter_pad (int, optional): Padding separating the images (default None).
        fill_value (int, optional): Fill value for inter-padding (default 0).
        preprocess (callable, optional): Pre processing to apply to the image 
            samples (default None).
        display (bool, optional): Display images (default False).
        save (bool, optional): Save images to disk (default True).
        sample_freq (int, optional): Frequency of samples (per sampler call) (default 1).
        

    """
    def __init__(self, name, directory='.', overwrite=False, 
                 view='torch', ext='.jpg', color=True, size=None,
                 inter_pad=None, fill_value=0, preprocess=None, 
                 display=False, save=True, sample_freq=1):
        self.name = name
        self.directory = process(directory, create=True)
        self.directory = path.join(self.directory, 'samples')
        self.directory = process(self.directory, create=True)
        self.overwrite = overwrite
        self.view = view
        self.ext = ext
        self.color = color
        self.size = size
        self.inter_pad = inter_pad
        self.fill_value = fill_value
        self.preprocess = preprocess
        self.display_samples = display
        self.save_samples = save
        self.sample_freq = sample_freq
        self.figure = None
        self.chkp = path.join(self.directory, ".{0}.chkp".format(self.name))
        self.counter, self.filename = torch.load(self.chkp) if path.exists(self.chkp) else (0, '')
    
    def sample(self, imgs):
        """Saves and/or displays functions depending on the configuration.
        
        Args:
            imgs (Tensor, Array, list or tuple): Image samples. Will
                automatically be put in a grid. Must be in the [0,1] range.
        """
        self.counter += 1
        if (self.save_samples or self.display_samples) \
            and (self.counter % self.sample_freq == 0):
            samples = self._get_sample(imgs)
            if self.save_samples:
                old_filename = self.filename
                new_filename = self._make_name()
                if new_filename == old_filename and not self.overwrite:
                    print('WARNING: Overwriting file in non overwrite mode.')
                elif self.overwrite:
                    try:
                        os.remove(old_filename)
                    except:
                        pass
                imwrite(new_filename, samples)
                torch.save((self.counter, new_filename), self.chkp)
            if self.display_samples:
                self.figure = imshow(samples, interactive=True,
                                     title=os.path.basename(self._make_name()), 
                                     figure=self.figure, view='cv')
        

    def __call__(self, imgs):
        """Same as :meth:`sample`"""
        self.sample(imgs)

    def _make_name(self):
        tmpl = '{0}_{1:07d}{2}'
        fname = tmpl.format(self.name, self.counter, self.ext)
        self.filename = path.join(self.directory, fname)
        return self.filename

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