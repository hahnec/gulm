import numpy as np
from scipy.interpolate import interp1d


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