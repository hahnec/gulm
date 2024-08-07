import numpy as np
from scipy.ndimage import gaussian_filter


def add_pala_noise(iq, clutter_db=-60, power=-2, impedance=.2, amp_culler_db=10, sigma=1.5, multi=1):

    awgn = awgn_noise(iq.size, power, impedance).reshape(*iq.shape) * multi
    swgn = awgn * iq.max() * 10**((amp_culler_db+clutter_db)/20)
    swgn[swgn>0] += iq.max() * 10**(clutter_db/20)
    swgn[swgn<0] -= iq.max() * 10**(clutter_db/20)
    iq_filt = gaussian_filter(swgn, sigma)
    iq_speckle = iq + iq_filt

    return iq_speckle

# np.mean([abs(awgn_noise(iq.size, power, impedance).reshape(*iq.shape) * iq.max() * 10**((amp_culler_db+clutter_db)/20)) for _ in range(128)], axis=0).std(axis=0)

# np.std(np.mean([awgn_noise(iq.size, power, impedance).reshape(*iq.shape) for _ in range(1)], axis=0), axis=0)
# np.std(np.mean([awgn_noise(iq.size, power, impedance).reshape(*iq.shape) for _ in range(16)], axis=0), axis=0)
# np.std(np.mean([awgn_noise(iq.size, power, impedance).reshape(*iq.shape)*4 for _ in range(16)], axis=0), axis=0)

def awgn_noise(length, power, bandwidth):
    """ https://dsp.stackexchange.com/questions/65975/gaussian-signal-generation """
    sigma = np.sqrt(bandwidth * 10**(power/10))
    noise = np.random.normal(0, sigma, length) 
    return noise
