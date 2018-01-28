import numpy as np

def clamped_gaussian(mean, std, min_value, max_value):
    if max_value <= min_value:
        return mean
    factor = 0.99
    while True:
        ret = np.random.normal(mean, std)
        if ret > min_value and ret < max_value:
            break
        else:
            std = std * factor
            ret = np.random.normal(mean, std)
        
    return ret

def exponential_size(val):
    return val * (np.exp(- np.random.uniform() ) ) / (np.exp(0) + 1)

# Accepts hwc-bgr image
def index_gauss(img, precision=None, crop_size=None, random_size=True, ratio=None, seed=None):
    """Returns indices (Numpy slice) of an image crop sampled spatially using a gaussian distribution.

    Args:
        img (Array): Image as a Numpy array (OpenCV view, hwc-BGR).
        precision (list or tuple, optional): Floats representing the precision
            of the Gaussians (default [1, 4])
        crop_size (list or tuple, optional): Ints representing the crop size
            (default [img_width/4, img_height/4]).
        random_size (bool, optional): If true, randomizes the crop size with
            a minimum of crop_size. It uses an exponential distribution such
            that smaller crops are more likely (default True).
        ratio (float, optional): Keep a constant crop ratio width/height (default None).
        seed (float, optional): Set a seed for np.random.seed() (default None)

    Note:
        - If `ratio` is None then the resulting ratio can be anything.
        - If `random_size` is False and `ratio` is not None, the largest dimension
          dictated by the ratio is adjusted accordingly:
                
                - `crop_size` is (w=100, h=10) and `ratio` = 9 ==> (w=90, h=10)
                - `crop_size` is (w=100, h=10) and `ratio` = 0.2 ==> (w=100, h=20)

    """
    np.random.seed(seed)
    dims = {"w": img.shape[1], "h": img.shape[0]}
    if precision is None:
        precision = {"w": 1, "h": 4}
    else:
        precision = {"w": precision[0], "h": precision[1]}
    
    if crop_size is None:
        crop_size = {key: int(dims[key] / 4) for key in dims}
    else:
        crop_size = {"w": crop_size[0], "h": crop_size[1]}

    if ratio is not None:
        ratio = max(ratio, 1e-4)
        if ratio > 1:
            if random_size:
                crop_size['h'] = int(max(crop_size['h'], exponential_size(dims['h']))) 
            crop_size['w'] = int(np.round(crop_size['h']*ratio))
        else:
            if random_size:
                crop_size['w'] = int(max(crop_size['w'], exponential_size(dims['w']))) 
            crop_size['h'] = int(np.round(crop_size['w']/ratio))
    else:
        if random_size:
            crop_size = {key: int(max(val, exponential_size(dims[key]))) for key, val in crop_size.items()}

    centers = {key: int(clamped_gaussian(dim / 2, crop_size[key] / precision[key], 
                                         min(int(crop_size[key] /2), dim), 
                                         max(int(dim - crop_size[key] / 2), 0))) 
                     for key, dim in dims.items()}
    starts = {key: max(center - int(crop_size[key] / 2), 0)
              for key, center in centers.items()}
    ends = {key: start + crop_size[key] for key, start in starts.items()}
    return np.s_[starts["h"]:ends["h"], starts["w"]:ends["w"], :]


def slice_gauss(img, precision=None, crop_size=None, random_size=True, ratio=None, seed=None):
    """Returns a cropped sample from an image array using :func:`index_gauss`"""
    return img[index_gauss(img, precision, crop_size, random_size, ratio)]


# Accepts hwc-bgr image
def index_uniform(img, crop_size=None, random_size=True, ratio=None, seed=None):
    """Returns indices (Numpy slice) of an image crop sampled spatially using a uniform distribution.

    Args:
        img (Array): Image as a Numpy array (OpenCV view, hwc-BGR).
        crop_size (list or tuple, optional): Ints representing the crop size
            (default [img_width/4, img_height/4]).
        random_size (bool, optional): If true, randomizes the crop size with
            a minimum of crop_size. It uses an exponential distribution such
            that smaller crops are more likely (default True).
        ratio (float, optional): Keep a constant crop ratio width/height (default None).
        seed (float, optional): Set a seed for np.random.seed() (default None)

    Note:
        - If `ratio` is None then the resulting ratio can be anything.
        - If `random_size` is False and `ratio` is not None, the largest dimension
          dictated by the ratio is adjusted accordingly:
                
                - `crop_size` is (w=100, h=10) and `ratio` = 9 ==> (w=90, h=10)
                - `crop_size` is (w=100, h=10) and `ratio` = 0.2 ==> (w=100, h=20)

    """
    np.random.seed(seed)
    dims = {"w": img.shape[1], "h": img.shape[0]}
    if crop_size is None:
        crop_size = {key: int(dims[key] / 4) for key in dims}
    if ratio is not None:
        ratio = max(ratio, 1e-4)
        if ratio > 1:
            if random_size:
                crop_size['h'] = int(max(crop_size['h'], exponential_size(dims['h']))) 
            crop_size['w'] = int(np.round(crop_size['h']*ratio))
        else:
            if random_size:
                crop_size['w'] = int(max(crop_size['w'], exponential_size(dims['w']))) 
            crop_size['h'] = int(np.round(crop_size['w']/ratio))
    else:
        if random_size:
            crop_size = {key: int(max(val, exponential_size(dims[key]))) for key, val in crop_size.items()}

    centers = {key: int(np.random.uniform(int(crop_size[key]/2), int( dims[key] - crop_size[key]/2) ))
                     for key, dim in dims.items()}
    starts = {key: max(center - int(crop_size[key] / 2), 0)
              for key, center in centers.items()}
    ends = {key: start + crop_size[key] for key, start in starts.items()}
    return np.s_[starts["h"]:ends["h"], starts["w"]:ends["w"], :]

def slice_uniform(img, crop_size=None, random_size=True, ratio=None, seed=None):
    """Returns a cropped sample from an image array using :func:`index_uniform`"""
    return img[index_uniform(img, crop_size, random_size, ratio)]