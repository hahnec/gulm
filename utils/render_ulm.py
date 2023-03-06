import sys
sys.path.append('../')

import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

from simple_tracker.tracking2d import tracking2d
from simple_tracker.tracks2img import tracks2img
from utils.srgb_conv import srgb_conv


normalize = lambda x: (x-x.min())/(x.max()-x.min()) if x.max()-x.min() > 0 else x-x.min()


def render_ulm(frames, tracking=None, expr='', plot_opt=False, cmap_opt=False, uint8_opt=False, gamma=None, srgb_opt=False, wavelength=None, size=None, origin=None, fps = 500):

    # init variables
    wavelength = 9.856e-05 if wavelength is None else wavelength
    origin = np.array([-72,  16, 0], dtype=int) if origin is None else origin
    size = np.array([84, 134]) if size is None else size

    # remove empty arrays
    frames = [f for f in frames if f.size > 0]

    if tracking == 'hungarian':
        # init variables
        min_len = 15
        max_linking_distance = 2
        max_gap_closing = 0

        # render based on Hungarian linker
        frames = [f / wavelength for f in frames]
        tracks_out, tracks_interp = tracking2d(frames, max_linking_distance=max_linking_distance, max_gap_closing=max_gap_closing, min_len=min_len, scale=1/fps, mode='interp')
        shifted_coords = [np.hstack([p[:, :2] - origin[:2], p[:, 2:]]) for p in tracks_out]
        ulm_img, vel_map = tracks2img(shifted_coords, img_size=size, scale=10, mode='tracks')#velnorm')
    else:
        # render based on localizations
        all_pts = np.vstack(frames) / wavelength - origin[:2]
        ulm_img, vel_map = tracks2img(all_pts, img_size=np.array([84, 134]), scale=10, mode='all_in')

    # gamma correction
    gamma = gamma if isinstance(gamma, (float, int)) else 1
    ulm_img **= gamma

    # sRGB conversion
    if srgb_opt: ulm_img = srgb_conv(normalize(ulm_img))

    # color mapping
    ulm_img = img_color_map(img=normalize(ulm_img), cmap='inferno')
    vel_map = img_color_map(img=normalize(vel_map), cmap='plasma')

    if plot_opt:
        plt.figure()
        plt.imshow(ulm_img)
        plt.show()

        plt.figure()
        plt.imshow(vel_map)
        plt.show()

    if uint8_opt:
        ulm_img = np.array(normalize(ulm_img) * (2**8-1), dtype='uint8')
        vel_map = np.array(normalize(vel_map) * (2**8-1), dtype='uint8')

    return ulm_img, vel_map

def img_color_map(img=None, cmap='inferno'):

    # get color map
    colormap = plt.get_cmap(cmap)

    # apply color map omitting alpha channel
    img = colormap(img)[..., :3]

    return img


def load_ulm_data(data_path, expr='pace'):

    # path management
    script_path = Path(__file__).resolve().parent.parent / 'scripts' / 'other_frames'
    data_path = data_path if data_path is not None else script_path
    fnames = sorted(Path(data_path).iterdir())
    assert len(fnames) > 0, 'No files found'

    # load frame data
    frames = []
    for fname in fnames:
        # skip files which do not contain expression
        if fname.name.__contains__(expr):
            arr = np.loadtxt(fname, delimiter=',', skiprows=1)
            if arr.size == 0: arr = np.ones([1, 2])*float('nan')     # change dimensions of empty array
            if len(arr.shape) != 2: arr = arr[None, :]
            frames.append(arr)
    assert len(frames) > 0, 'No frames found'

    return frames


if __name__ == '__main__':

    data_path = None
    frames = load_ulm_data(data_path, expr='pala')
    render_ulm(frames, plot_opt=True, gamma=.9, srgb_opt=True)
    render_ulm(frames, tracking='hungarian', plot_opt=True, gamma=.9, srgb_opt=True)
