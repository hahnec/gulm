import numpy as np
from scipy import signal


def bandpass_filter(channel_data, freq_cen=None, freq_smp=None):
    """
    automatic bandpass-filtering around prevalent frequency component
    """

    # detect relative frequency
    freq_cen = detect_freq(channel_data) if freq_cen is None else freq_cen

    if freq_cen <= 0:
        warnings.warn('Skip bandpass filter due to invalid frequency')
        return channel_data

    # set cut-off frequencies (where amplitude drops by 3dB)
    sw = 0.5
    lo, hi = np.array([sw, (2-sw)]) * freq_cen
    lo, hi = max(0, lo), min(2*freq_cen-np.spacing(1), hi)

    sos = signal.butter(5, [lo, hi], btype='band', output='sos', fs=freq_smp)
    y = signal.sosfiltfilt(sos, channel_data, axis=0)

    return y


def detect_freq(channel_data):

    w = np.fft.fft(channel_data)
    freqs = np.fft.fftfreq(len(w))
    idx = np.argmax(np.abs(w))
    freq = abs(freqs[idx]) * 2

    return freq
