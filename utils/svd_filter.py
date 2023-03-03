import numpy as np

def svd_filter(iq, cutoff=4):

    init_shape = iq.shape

    # reshape into Casorati matrix
    x = iq.reshape(-1, iq.shape[-1], order='F')

    # calculate svd of the autocorrelated Matrix
    u, _, _ = np.linalg.svd(x.T@x, full_matrices=True)

    # calculate the singular vectors
    v = x@u

    # singular value decomposition
    n = v[:, cutoff:]@u[:, cutoff:].T

    # Reconstruction of the final filtered matrix
    iqf = n.reshape(init_shape, order='F')   

    return iqf