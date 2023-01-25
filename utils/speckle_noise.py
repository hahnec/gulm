import numpy as np
from scipy.ndimage import gaussian_filter

def pala_add_noise_iq(iq, clutter_db=-60, power=-2, impedance=.2, amp_culler_db=10, sigma=1.5):

    iq = abs(iq)
    awgn = awgn_noise(iq.size,1,power,impedance).reshape(*iq.shape)
    swgn = awgn*max(iq)*10**((amp_culler_db+clutter_db)/20)
    iq_filt = gaussian_filter(max(iq)*10**(clutter_db/20) * swgn, sigma)
    iq_speckle = iq + iq_filt

    return iq_speckle


def awgn_noise(length, power, Bandwidth):
    """ https://dsp.stackexchange.com/questions/65975/gaussian-signal-generation """
    sigma = np.sqrt(Bandwidth * 10**(power/10))
    noise = np.random.normal(0, sigma, length) 
    return noise
