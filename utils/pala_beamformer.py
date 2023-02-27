# beamforming with data explained at https://github.com/CorazzaAlexandre/PALA_Beamforming

import numpy as np
#from scipy.interpolate import interp1d


bf_demod_100_bw2iq = lambda rf_100bw: rf_100bw[0::2, ...] - 1j*rf_100bw[1::2, ...]


def decompose_frame(rf_frame, angle_num, sample_num):

    rf_iq_frame = []
    # iterate over number of transmitted angles
    for i_tx in range(angle_num):

        # decompose stacked channels in sample domain for all angles
        idx_list = i_tx*sample_num + np.arange(sample_num)
        rf_i = rf_frame[idx_list, ...]

        # convert 100 bandwidth data to IQ signal ?
        rf_iq = bf_demod_100_bw2iq(rf_i)

        rf_iq_frame.append(rf_iq)

    return np.array(rf_iq_frame)

def pala_beamformer(rf_iq_frame, param, mesh_x, mesh_z):

    # Delay And Sum beamforming
    iq_frame = bf_das(rf_iq_frame, param, mesh_x, mesh_z)

    # rescale and remove imaginary components
    bmode = 20*np.log10(abs(iq_frame), where=abs(iq_frame)>0)

    # remove infs and NaNs
    bmode[np.isnan(bmode) | np.isinf(bmode)] = np.min(bmode[np.isfinite(bmode)])

    return bmode


def bf_das(rf_iq, param, x, z):

    iq_frame = np.zeros_like(x, dtype=rf_iq.dtype)

    if not hasattr(param, 'compound'):
        param.compound = 1

    # iterate over number of transmitted angles
    for k in range(len(param.angles_list)):
        param.theta = param.angles_list[k]
        # rf_iq dimensions: angles x samples x channels
        rf_angle = rf_iq[k, ...]
        if param.compound:
            # accumulate receiver delay and sum
            iq_frame += bf_das_rx(rf_angle, param, x, z)
        else:
            # attach receiver delay and sum
            iq_frame[..., k] = bf_das_rx(rf_angle, param, x, z)

    return iq_frame


def bf_das_rx(sig, param, x, z):

    agg_sig = np.zeros([1, x.size], dtype=sig.dtype)

    # emit delay
    # TXdelay = (1/param.c)*tan(param.theta)*abs(param.xe - param.xe(1));

    # virtual source (non-planar wave assumption)
    beta = 1e-8
    width = param.xe[-1]-param.xe[0]    # extent of the phased-array
    vsource = [-width*np.cos(param.theta) * np.sin(param.theta)/beta, -width*np.cos(param.theta)**2/beta]

    # iterate over channels
    for k in range(param.Nelements):
        # dtx = sin(param.theta)*X(:)+cos(param.theta)*Z(:); %convention FieldII
        # dtx = sin(param.theta)*X(:)+cos(param.theta)*Z(:) + mean(TXdelay)*param.c; %convention FieldII
        # dtx = sin(param.theta)*X(:)+cos(param.theta)*Z(:) + mean(TXdelay-min(TXdelay))*param.c; %convention Verasonics

        # find transmit travel distances considering virtual source
        dtx = np.hypot(x.T.flatten()-vsource[0], z.T.flatten()-vsource[1]) - np.hypot((abs(vsource[0])-width/2)*(abs(vsource[0])>width/2), vsource[1])

        # find receive travel distances
        drx = np.hypot(x.T.flatten()-param.xe[k], z.T.flatten())

        # convert overall travel distances to delay times
        tau = (dtx+drx) / param.c

        # convert phase-shift delays into sample indices (deducting blind zone?)
        idxt = (tau-param.t0) * param.fs #+ 1
        I = ((idxt<1) + (idxt>sig.shape[0]-1)).astype(bool)
        idxt[I] = 1 # arbitrary index, will be soon rejected

        idx = idxt       # floating number representing sample index for subsequent interpolation
        idxf = np.floor(idx).astype('int64') # rounded number of samples
        IDX = idx #np.repmat(idx, [1 1 sig.shape[2]]) #3e dimension de SIG: angles

        # resample at delayed sample positions (using linear interpolation)
        #f = interp1d(np.arange(sig.shape[0]), sig[..., k])
        #temp = f(idxt)
        #temp = sig[idxf, k].flatten()
        temp = sig[idxf, k].T.flatten() * (idxf+1-IDX) + sig[idxf+1, k].T.flatten() * (IDX-idxf)
        #temp = sig[idxf-1, k].T.flatten() * (idxf+1-IDX) + sig[idxf, k].T.flatten() * (IDX-idxf)

        # mask values outside index range
        temp[I] = 0

        # IQ to RF conversion
        if np.any(~np.isreal(temp)):
            temp = temp * np.exp(2*1j*np.pi*param.f0*tau)

        # F-number mask
        mask_Fnumber = abs(x-param.xe[k]) < z / param.fnumber/2

        # sum delayed channel signal
        agg_sig += temp.T.flatten() * mask_Fnumber.T.flatten()

    output = agg_sig.reshape(x.shape, order='F')

    return output
