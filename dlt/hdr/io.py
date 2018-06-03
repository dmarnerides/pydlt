import os
import sys
import numpy as np
import cv2
from ..util import rgb2bgr
from ..util.paths import process

# Accepts hwc - BGR float32 numpy array (cv style)
# TODO: Improve multiple views support
# TODO: Improve speed
def write_pfm(filename, img, scale=1):
    """Writes an OpenCV image into pfm format on disk.

    Args:
        filename (str): Name of the image file. The .pfm extension is not added.
        img (Array): Numpy Array containing the image (OpenCV view hwc-BGR)
        scale (float): Scale factor for file. Positive for big endian,
            otherwise little endian. The number tells the units of the samples
            in the raster (default 1)
    """
    if img.dtype.name != 'float32':
        raise TypeError('Image dtype must be float32.')

    with open(filename, 'w') as file:
        file.write('PF\n' if img.shape[2] == 3 else 'Pf\n')
        file.write('{w} {h}\n'.format(w=img.shape[1], h=img.shape[0]))

        endian = img.dtype.byteorder

        if endian == '<' or endian == '=' and sys.byteorder == 'little':
            scale = -scale

        file.write('%f\n' % scale)
        img = np.flip(np.flip(img, 2), 0)
        img.tofile(file)

# returns the image in hwc - BGR (cv style)
# TODO: Improve multiple views support
# TODO: Improve speed
def load_pfm(filename):
    """Loads a pfm image file from disk into a Numpy Array (OpenCV view).

    Supports HDR and LDR image formats.
    
    Args:
        filename (str): Name of pfm image file.
    """
    filename = process(filename)
    with open(filename, "r", encoding="ISO-8859-1") as file:
        nc = 3 if file.readline().rstrip() == "PF" else 1
        width, height = [int(x) for x in file.readline().rstrip().split()]
        shape = (height, width, nc)
        img = np.fromfile(file, '{0}{1}'.format("<" if float(file.readline().rstrip()) < 0 else ">",'f') )
        img = np.reshape(img, shape)
        return np.flip(np.flip(img, 2), 0).copy()

def load_dng(filename, **kwargs):
    """Loads a dng image file from disk into a float32 Numpy Array (OpenCV view).

    Requires rawpy.

    Args:
        filename (str): Name of pfm image file.
        **kwargs: Extra keyword arguments to pass to `rawpy.postprocess()`.
    """
    import rawpy
    filename = process(filename)
    with rawpy.imread(filename) as raw:
        default_kwargs = dict(gamma=(1,1), no_auto_bright=True, output_bps=16)
        default_kwargs.update(kwargs)
        img = raw.postprocess(**default_kwargs)
    return rgb2bgr(img,-1).astype('float32')

# Accepts hwc - BGR float32 numpy array (cv style)
# TODO: Improve multiple views support
# TODO: Improve speed
def imwrite(filename, img, *args, **kwargs):
    """Writes an image to disk. Supports HDR and LDR image formats.

    Args:
        filename (str): Name of image file.
        img (Array): Numpy Array containing the image (OpenCV view hwc-BGR).
        *args: Extra arguments to pass to cv2.imwrite or write_pfm if saving a
            .pfm image.
        **kwargs: Extra keyword arguments to pass to cv2.imwrite or write_pfm
            if saving a .pfm image.
    """
    ext = os.path.splitext(filename)[1]
    if ext.lower() == '.pfm':
        write_pfm(filename, img, *args, **kwargs)
    else:
        cv2.imwrite(filename, img, *args, **kwargs)

def imread(filename):
    """Reads an image file from disk into a Numpy Array (OpenCV view).

    Args:
        filename (str): Name of pfm image file.
    """
    filename = process(filename)
    ext = os.path.splitext(filename)[1]
    if ext.lower() == '.pfm':
        return load_pfm(filename)
    elif ext.lower() == '.dng':
        return load_dng(filename)
    else:
        loaded = cv2.imread(filename, flags=cv2.IMREAD_ANYDEPTH + cv2.IMREAD_COLOR)
        if loaded is None:
            raise IOError('Could not read {0}'.format(filename))
        else:
            return loaded

def load_encoded(filename):
    """Loads a file as a Numpy Byte (uint8) Array.

    Args:
        filename (str): Name of file.
    """
    return np.fromfile(filename, dtype='uint8')

def decode_loaded(x):
    """Decodes an image stored in a Numpy Byte (uint8) Array using OpenCV.

    Args:
        x: The Numpy Byte (uint8) Array.
    """
    return cv2.imdecode(x, flags=cv2.IMREAD_ANYDEPTH + cv2.IMREAD_COLOR)