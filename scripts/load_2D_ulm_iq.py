import numpy as np
import scipy.io
from scipy.interpolate import interp1d
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from omegaconf import OmegaConf
from scipy import signal

from utils.pala_beamformer import pala_beamformer
from utils.svd_filter import svd_filter


bf_demod_100_bw2iq = lambda rf_100bw: rf_100bw[:, 0::2, ...] - 1j*rf_100bw[:, 1::2, ...]

def iq2rf(iq_data, mod_freq, upsample_factor=10):

    x = np.linspace(0, len(iq_data)/mod_freq, num=len(iq_data), endpoint=True)
    t = np.linspace(0, len(iq_data)/mod_freq, num=int(len(iq_data)*upsample_factor), endpoint=True)
    f = interp1d(x, iq_data, axis=0)
    y = f(t)

    # take 2nd dimension into account
    t = t[:, None] if len(iq_data.shape) == 2 else t

    rf_data = y * np.exp(2*1j*np.pi*mod_freq*t)

    rf_data = 2**.5 * rf_data.real

    return rf_data


script_path = Path(__file__).parent.resolve()

# load config
cfg = OmegaConf.load(str(script_path.parent / 'config_invivo.yaml'))

# override config with CLI
cfg = OmegaConf.merge(cfg, OmegaConf.from_cli())

rel_path = Path(cfg.loc_dir).parent / 'PALA_data_InVivoRatBrain'
seq_fname = rel_path / 'Rat18_2D_PALA_0323_163558_sequence.mat'

rf_fname = rel_path / 'IQ' / ('PALA_InVivoRatBrain_'+str(cfg.dat_num).zfill(3)+'.mat')
rf_mat = scipy.io.loadmat(rf_fname)
RFdata = rf_mat['IQ']

# test svd filter performance
A = np.array([[[ 1.0933 - 0.1924j, -0.8637 - 0.7648j, -1.2141 - 1.4224j],
               [ 1.1093 + 0.8886j,  0.0774 - 1.4023j, -1.1135 + 0.4882j]],

              [[-0.0068 - 0.1774j, -0.7697 + 1.4193j, -0.2256 + 0.1978j],
               [ 1.5326 - 0.1961j,  0.3714 + 0.2916j,  1.1174 + 1.5877j]],

              [[-1.0891 - 0.8045j,  0.5525 + 0.8351j,  1.5442 + 0.2157j],
               [ 0.0326 + 0.6966j,  1.1006 - 0.2437j,  0.0859 - 1.1658j]],

              [[-1.4916 - 1.1480j, -1.0616 + 0.7223j, -0.6156 - 0.6669j],
               [-0.7423 + 0.1049j,  2.3505 + 2.5855j,  0.7481 + 0.1873j]]]).swapaxes(0, 1).swapaxes(1, -1)

assert np.allclose(A-svd_filter(A, cutoff=0), 0), 'SVD without cutoff should yield same as input'

bmodes_svd = svd_filter(RFdata, cutoff=4)

but_b,but_a = signal.butter(2, np.array([50, 250])/(1000/2), btype='bandpass')
bmodes_filt = signal.filtfilt(but_b, but_a, bmodes_svd, axis=2)

# extract video
import cv2
fps = 30
fourcc = cv2.VideoWriter_fourcc(*'RGBA')
out = cv2.VideoWriter('bmode_svd_filt.avi', fourcc, fps, (bmodes_filt.shape[1], bmodes_filt.shape[0]), False)
bmodes_out = abs(bmodes_filt)
bmodes_out[np.isinf(bmodes_out) | np.isnan(bmodes_out)] = 0
bmodes_out = (bmodes_out-bmodes_out.min())/(bmodes_out.max()-bmodes_out.min())
bmodes_out = np.round(bmodes_out * 255).astype(np.uint8)
for i in range(bmodes_out.shape[2]):
    out.write(bmodes_out[..., i])
out.release()

bmodes = RFdata.swapaxes(1, 2).swapaxes(0, 1)
bmodes_svd = bmodes_svd.T.swapaxes(1, -1)
bmodes_filt =  bmodes_filt.T.swapaxes(1, -1)

# rescale for plot
bmodes[bmodes == 0] = np.spacing(1)
bmodes[np.isinf(bmodes) | np.isnan(bmodes)] = 0
bmodes = 20*np.log10(abs(bmodes))
bmodes = bmodes-bmodes.max()

# rescale for plot
bmodes_svd[bmodes_svd == 0] = np.spacing(1)
bmodes_svd[np.isinf(bmodes_svd) | np.isnan(bmodes_svd)] = 0
bmodes_svd = 20*np.log10(abs(bmodes_svd))
bmodes_svd = bmodes_svd-bmodes_svd.max()

# rescale for plot
bmodes_filt[bmodes_filt == 0] = np.spacing(1)
bmodes_filt[np.isinf(bmodes_filt) | np.isnan(bmodes_filt)] = 0
bmodes_filt = 20*np.log10(abs(bmodes_filt))
bmodes_filt = bmodes_filt-bmodes_filt.max()

fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(30/3, 15/3))
bmode_limits = [-30, 0]
im1 = axs[0].imshow(bmodes[19], vmin=bmode_limits[0], vmax=bmode_limits[1], interpolation='none', cmap='gray')
im2 = axs[1].imshow(bmodes_svd[19], vmin=bmode_limits[0], vmax=bmode_limits[1], interpolation='none', cmap='gray')
im3 = axs[2].imshow(bmodes_filt[19], vmin=bmode_limits[0], vmax=bmode_limits[1], interpolation='none', cmap='gray')
plt.savefig('bmode_svd_filt.png')
plt.show()



def animate_func_bmode(i):
    if i % fps == 0:
        print( '.', end ='' )

    im1.set_array(bmodes[i])
    im2.set_array(bmodes_svd[i])
    im3.set_array(bmodes_filt[i])
    return [im1, im2, im3]


fps = 25
nSeconds = len(bmodes) // fps

import matplotlib.animation as animation
anim = animation.FuncAnimation(
                            fig, 
                            animate_func_bmode, 
                            frames = nSeconds * fps,
                            interval = 1000 / fps, # in ms
                            )

anim.save('bmode_anim.mp4', fps=fps, extra_args=['-vcodec', 'libx264'])