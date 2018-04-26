import warnings
import matplotlib.pyplot as plt
from ..util import to_array, change_view, make_grid

## https://stackoverflow.com/questions/22873410/how-do-i-fix-the-deprecation-warning-that-comes-with-pylab-pause
warnings.filterwarnings("ignore", ".*GUI is implemented.*")

def imshow(img, view='torch', figure=None, pause=0, title=None, *args, **kwargs):
    """Displays a Tensor or Array image to screen.

    Args:
        img (Tensor or Array): Image to display.
        view (str, optional): View of image. For more details see
            :func:`dlt.util.change_view` (default 'torch').
        figure (int, optional): Use selected figure (default None).
        pause (float, optional): Number of seconds to pause execution for
            displaying. If 0 execution is stopped indefinitely. Useful for
            displaying changing images in a video-like fashion (default 0).
        title (str, optional): Title for figure (default None).
        *args (optional): Extra arguments to be passed to plt.imshow().
        **kwargs (optional): Extra keyword arguments to be passed to plt.imshow().
    Example:
        >>> for video_1_frame, video_2_frame in two_videos_frames:
        >>>     dlt.viz.imshow(video_1_frame, view='cv', figure=1, pause=0.2, title='Video 1')
        >>>     dlt.viz.imshow(video_2_frame, view='cv', figure=2, pause=0.2, title='Video 2')

    """
    if figure is None:
        if imshow.my_figure is None:
            imshow.my_figure = plt.figure().number
        figure = imshow.my_figure
    else:
        figure = plt.figure(figure).number

    if title is not None:
        f = plt.gcf()
        f.canvas.set_window_title(title)
    if pause > 0:
        plt.ion()
        plt.clf()
    else:
        plt.ioff()
        
    img = to_array(img)
    if img.ndim not in (2,3):
        raise ValueError('Images must have two or three dimensions.')

    img = change_view(img, view, 'plt')
    if img.shape[-1] not in (1,3,4):
        raise ValueError('Invalid number of channels ({0}). '.format(img.shape[-1])
                         + 'Perhaps you used the wrong view?')
    img = img.squeeze()
    plt.imshow(img, cmap='gray' if img.ndim == 2 else None, *args, **kwargs)
    if pause > 0:
        plt.pause(pause)
    else:
        plt.show()

imshow.my_figure = None
