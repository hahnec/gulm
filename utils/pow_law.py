import numpy as np

pow_law_fun = lambda x, a=140.1771, b=1.1578: a*x**-b


def compensate_pow_law(data=None, x=None, a=140.1771, b=1.1578, c=343, fkHz=175, sample_rate=1.):

    # compute sample positions in millimeter distances
    if x is None:
        x = sample2dist(np.arange(len(self.hilbert_data)) + np.spacing(1), c=c, fkHz=fkHz, sample_rate=sample_rate)

    denom = pow_law_fun(x, a=a, b=b)
    denom = denom[..., None] if len(data.shape)-1 == len(denom.shape) else denom
    ouptut = data / denom

    return ouptut