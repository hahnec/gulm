import scipy.io
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from utils.pala_beamformer import pala_beamformer, decompose_frame
from utils.iq2rf import iq2rf


rf_fname = '/home/chris/LocalDatasets/03_PALA/PALA_data_InSilicoFlow/RF/PALA_InSilicoFlow_RF008.mat'
rf_mat = scipy.io.loadmat(rf_fname)

iq_fname = '/home/chris/LocalDatasets/03_PALA/PALA_data_InSilicoFlow/IQ/PALA_InSilicoFlow_IQ008.mat'
iq_mat = scipy.io.loadmat(iq_fname)

cfg_fname = '/home/chris/LocalDatasets/03_PALA/PALA_data_InSilicoFlow/PALA_InSilicoFlow_v3_config.mat'
cfg_mat = scipy.io.loadmat(cfg_fname)

seq_fname = '/home/chris/LocalDatasets/03_PALA/PALA_data_InSilicoFlow/PALA_InSilicoFlow_sequence.mat'
seq_mat = scipy.io.loadmat(seq_fname)

pos_fname = '/home/chris/LocalDatasets/03_PALA/PALA_data_InSilicoFlow/PALA_InSilicoFlow_v3_pos_Tracks_dt.mat'
pos_mat = scipy.io.loadmat(pos_fname)

res_fname = '/home/chris/LocalDatasets/03_PALA/PALA_data_InSilicoFlow/Results/PALA_InSilicoFlow_MatOut_multi_10dB.mat'
res_mat = scipy.io.loadmat(res_fname)

frame_idx = 0

gt_data = pos_mat['Tracks_dt'][:, 0]

max_time = max([np.max(scatterer[:, 3]) for scatterer in gt_data])
min_time = min([np.min(scatterer[:, 3]) for scatterer in gt_data])

time_series = []
for i in range(int(min_time), int(max_time), 150):
    scatterers_t = np.array([scatterer[scatterer[:, 3] == i][0] for scatterer in gt_data if np.any(scatterer[:, 3] == i)])
    time_series.append(scatterers_t)

mat2dict = lambda mat: dict([(k[0], v.squeeze()) for v, k in zip(mat[0][0], list(mat.dtype.descr))])

P = mat2dict(seq_mat['P'])
PData = mat2dict(seq_mat['PData'])
Trans = mat2dict(seq_mat['Trans'])
Media = mat2dict(seq_mat['Media'])
UF = mat2dict(seq_mat['UF'])
Resource = mat2dict(seq_mat['Resource'])
TW = mat2dict(seq_mat['TW'])
TX = mat2dict(seq_mat['TX'])
Receive = mat2dict(seq_mat['Receive'])

RFdata = rf_mat['RFdata']
ListPos = rf_mat['ListPos']
Media = mat2dict(rf_mat['Media'])
P = mat2dict(rf_mat['P'])

class Param:
    pass

param = Param()
param.bandwidth = (Trans['Bandwidth'][1] - Trans['Bandwidth'][0]) / Trans['frequency'] * 100
param.f0 = Trans['frequency']*1e6 #central frequency [Hz]
param.fs = Receive['demodFrequency']*1e6 # sampling frequency (100% bandwidth mode of Verasonics) [Hz]
param.c = float(Resource['Parameters']['speedOfSound']) # speed of sound [m/s]
param.wavelength = param.c / param.f0 # Wavelength [m]
param.xe = Trans['ElementPos'][:, 0]/1000 # x coordinates of transducer elements [m]
param.Nelements = Trans['numelements'] # number of transducers
blind_zone = P['startDepth']*param.wavelength # minimum z in [m]
param.t0 = 2*blind_zone/param.c - TW['peak']/param.f0 # time between the emission and the beginning of reception [s]

angles_list = np.array([TX['Steer']*1, TX['Steer']*0, TX['Steer']*-1, TX['Steer']*0])
angles_list = angles_list[:P['numTx'], 0]
param.angles_list = angles_list # list of angles [rad] (in TX.Steer)
param.fnumber = 1.9 # fnumber
param.compound = 1 # flag to compound [1/0]

# pixel grid (extracted from PData), in [m] 
param.x = (PData['Origin'][0]+np.arange(PData['Size'][1])*PData['PDelta'][2])*param.wavelength
param.z = (PData['Origin'][2]+np.arange(PData['Size'][0])*PData['PDelta'][0])*param.wavelength
[mesh_x, mesh_z] = np.meshgrid(param.x, param.z)
del angles_list

extent = [min(param.x), max(param.x), min(param.z), max(param.z)]
aspect = max(param.z)/(max(param.x)-min(param.x))

# iterate over frames
iters = 1
for frame_idx in range(iters):
    # rf_iq_frame dimensions: angles x samples x channels
    rf_iq_frame = decompose_frame(P, RFdata[..., frame_idx])
    bmode = pala_beamformer(rf_iq_frame, param, mesh_x, mesh_z)
    bmode -= bmode.max()

bmode_limits = [-60, 0] # [dB] scale

# ground truth
xpos = ListPos[:, 0, 0] * param.wavelength
zpos = ListPos[:, 2, 0] * param.wavelength

enlarge_factor = 100
rescale = len(param.z)*enlarge_factor / rf_iq_frame.shape[1]

# virtual source (non-planar wave assumption)
beta = 1e-8
width = param.xe[-1]-param.xe[0]    # extent of the phased-array
vsource = [-width*np.cos(param.angles_list[1]) * np.sin(param.angles_list[1])/beta, -width*np.cos(param.angles_list[1])**2/beta]

# select channel and target
ch_idx = 74-50#118
ch_gap = ch_idx + 5#-4
for pt_idx in range(15, ListPos.shape[0]):

    if np.isnan(xpos[pt_idx]) or np.isnan(zpos[pt_idx]) is np.nan:
        continue

    fig = plt.figure(figsize=(30, 15))
    gs = gridspec.GridSpec(2, 2)
    ax1 = plt.subplot(gs[:, 0])
    ax2 = plt.subplot(gs[0, 1])
    ax3 = plt.subplot(gs[1, 1])

    ax1.imshow(bmode[::-1, ...], vmin=bmode_limits[0], vmax=bmode_limits[1], extent=extent, aspect=aspect**-1, cmap='gray')
    ax1.set_facecolor('#000000')
    ax1.plot(xpos, zpos, 'rx', label='ground-truth')
    ax1.plot([min(param.x), max(param.x)], [0, 0], color='gray', linewidth=5, label='Transducer plane')
    ax1.set_ylim([0, max(param.z)])
    ax1.set_xlim([min(param.x), max(param.x)])
    ax1.set_xlabel('Lateral domain $x$ [m]')
    ax1.set_ylabel('Axial distance $z$ [m]')

    # find transmit travel distances considering virtual source
    dtx = np.hypot(xpos[pt_idx]-vsource[0], zpos[pt_idx]-vsource[1]) - np.hypot((abs(vsource[0])-width/2)*(abs(vsource[0])>width/2), vsource[1])

    for ax, el_idx, color in zip([ax2, ax3], [ch_idx, ch_gap], ['b', 'g']):

        # find receive travel distances
        drx = np.hypot(xpos[pt_idx]-param.xe[el_idx], zpos[pt_idx])

        # convert overall travel distances to travel times
        tau = (dtx+drx) / param.c

        # convert travel times to sample indices (deducting blind zone?)
        idxt = (tau-param.t0) * param.fs
        distance = idxt * enlarge_factor

        # add undetected range
        raw_channel = rf_iq_frame[1, :, el_idx]
        channel = iq2rf(raw_channel, mod_freq=param.f0, upsample_factor=enlarge_factor)
        ax.plot(channel, label='Receive element %s' % el_idx, color=color)
        ax.plot([distance, distance], [min(channel), max(channel)], label='Point number: %s' % pt_idx, color='r')
        #ax.set_xlim([0, max(param.z)])
        ax.set_xlabel('Axial distance $z$ [m]')
        ax.set_ylabel('Amplitude $A(z)$ [a.u.]')
        ax.grid(True)
        ax.legend()

        # plot rx trajectory
        ax1.plot([param.xe[el_idx], xpos[pt_idx]], [0, zpos[pt_idx]], color, linestyle='dashed', label='receive trajectory of element %s' % el_idx)
        ax1.legend()

    plt.show()