import sys

sys.path.append('../')

import time
from pathlib import Path

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import scipy.io
import torch
import wandb
import warnings
from matplotlib.patches import Ellipse
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
from mpl_toolkits.axes_grid1.inset_locator import mark_inset, zoomed_inset_axes
from omegaconf import OmegaConf
from scipy import signal
from skimage.metrics import structural_similarity
from sklearn.cluster import MeanShift, estimate_bandwidth

from gte_intersect.ellipse import EllipseIntersection
from multimodal_emg.batch_staged_memgo import batch_staged_memgo
from utils.bandpass import bandpass_filter
from utils.iq2rf import iq2rf
from utils.pala_beamformer import decompose_frame, pala_beamformer
from utils.pala_error import rmse_unique
from utils.render_ulm import load_ulm_data, render_ulm
from utils.speckle_noise import add_pala_noise
from utils.svd_filter import svd_filter
from utils.pow_law import compensate_pow_law

bf_demod_100_bw2iq = lambda rf_100bw: rf_100bw[:, 0::2, ...] - 1j*rf_100bw[:, 1::2, ...]


def angle_amplitude_ratio(amplitude, xe_pos, pt):
    
    tx_pos = np.array([xe_pos, 0])

    # incidence angle w.r.t. perpendicular
    angle = np.arccos(np.array([0, 1]) @ np.array(pt-tx_pos) / (np.linalg.norm([0, 1]) * np.linalg.norm(pt-tx_pos)))#*180/np.pi

    # ratio with angle 
    fov_ratio = amplitude / (abs(angle)/np.pi*2+1)    # denom in [1, +inf] with orthogonal incidence being 1

    return fov_ratio, angle

def get_amp_grad(data, toa, phi_shift, ch_idx, grad_step=1):

    # get sample index
    sample_idx = time2sample(toa, phi_shift)
    # ensure sample index is within boundaries
    sample_idx[(sample_idx >= len(data)) | (sample_idx < 0)] = 0
    # get sample amplitude
    sample_amp = data[sample_idx, ch_idx]
    # ensure adjacent sample index within boundaries
    sample_idx[(sample_idx >= len(data)-grad_step) | (sample_idx+grad_step < 0)] = 0
    # get sample amplitude
    gradie_amp = data[sample_idx+grad_step, ch_idx] - sample_amp

    return sample_amp, gradie_amp

def get_overall_phase_shift(data_arr, toas, phi_shifts, ch_idx, cch_sample, cch_grad, grad_step=1):
    
    phi_shifts[-np.pi > phi_shifts] += +2*np.pi
    phi_shifts[+np.pi < phi_shifts] += -2*np.pi

    sample, grad = get_amp_grad(data_arr, toas, phi_shifts, ch_idx, grad_step)
    phi_shifts[(sample < 0) & (cch_sample > 0) & (grad > 0) & (cch_grad < 0)] -= np.pi/2     
    phi_shifts[(sample > 0) & (cch_sample < 0) & (grad < 0) & (cch_grad > 0)] += np.pi/2

    sample, grad = get_amp_grad(data_arr, toas, phi_shifts, ch_idx, grad_step)
    phi_shifts[(sample < 0) & (cch_sample > 0) & (((grad > 0) & (cch_grad > 0)) | ((grad < 0) & (cch_grad < 0)))] += np.pi/2
    phi_shifts[(sample > 0) & (cch_sample < 0) & (((grad > 0) & (cch_grad > 0)) | ((grad < 0) & (cch_grad < 0)))] -= np.pi/2

    sample, grad = get_amp_grad(data_arr, toas, phi_shifts, ch_idx, grad_step)
    phi_shifts[(sample < 0) & (cch_sample < 0) & (grad > 0) & (cch_grad < 0)] += np.pi/2
    phi_shifts[(sample < 0) & (cch_sample < 0) & (grad < 0) & (cch_grad > 0)] -= np.pi/2

    sample, grad = get_amp_grad(data_arr, toas, phi_shifts, ch_idx, grad_step)
    phi_shifts[(sample > 0) & (cch_sample > 0) & (grad > 0) & (cch_grad < 0)] -= np.pi/2
    phi_shifts[(sample > 0) & (cch_sample > 0) & (grad < 0) & (cch_grad > 0)] += np.pi/2

    # take full lambda phase shift into account
    phi_shifts[-np.pi > phi_shifts] += +2*np.pi
    phi_shifts[+np.pi < phi_shifts] += -2*np.pi

    return phi_shifts

script_path = Path(__file__).parent.resolve()

# load config
cfg = OmegaConf.load(str(script_path.parent / 'config_invivo.yaml'))

# override config with CLI
cfg = OmegaConf.merge(cfg, OmegaConf.from_cli())

assert (1000/cfg.frame_batch_size)%1 == 0, 'frame_batch_size must be multiple of 1000'

if cfg.logging: 
    wandb.init(project="pulm", name=None, config=cfg, group=None)
    wandb.define_metric('PULM/MeanFrameConfidence', step_metric='frame')

run_name = str(wandb.run.name) if cfg.logging else 'wo_logging'
output_path = Path(cfg.data_dir) / 'Results' / ('invivo_frames_'+run_name)
if cfg.save_opt and not output_path.exists(): output_path.mkdir()

if cfg.plt_comp_opt or cfg.plt_frame_opt:
    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "Helvetica"
    })

rel_path = Path(cfg.data_dir)

# initialize intersector
ell_intersector = EllipseIntersection()

time2sample = lambda toa, phi_shift: np.round(((toa-nonplanar_tdx - phi_shift/(2*np.pi*param.fs) * param.c)/param.c - param.t0) * param.fs * cfg.enlarge_factor).astype(int)

torch.cuda.empty_cache()
frame_batch_size = cfg.frame_batch_size

results_path = Path(cfg.data_dir) / 'Results' / 'PALA_InVivoRatBrain_MatOut_multi.mat'
out_mat = scipy.io.loadmat(str(results_path))

blind_zone_idx = 0 #(1500//cfg.enlarge_factor)#3*(1500//cfg.enlarge_factor)//2

acc_pace_errs = []
acc_pala_errs = []
for dat_num in range(1, cfg.dat_num+1):

    seq_fname = rel_path / 'Rat18_2D_PALA_0323_163558_sequence.mat'
    seq_mat = scipy.io.loadmat(seq_fname)

    rf_fname = rel_path / 'RFdata' / ('Rat18_2D_PALA_0323_163558RF'+str(dat_num).zfill(3)+'.mat')
    rf_mat = scipy.io.loadmat(rf_fname)

    iq_fname = rel_path / 'IQ' / ('PALA_InVivoRatBrain_'+str(dat_num).zfill(3)+'.mat')
    iq_mat = scipy.io.loadmat(iq_fname)['IQ']
    iq_mat = svd_filter(iq_mat, cutoff=4)
    but_b,but_a = signal.butter(2, np.array([50, 250])/(1000/2), btype='bandpass')
    iq_mat = signal.filtfilt(but_b, but_a, iq_mat, axis=2)
    iq_mat[iq_mat == 0] = np.spacing(1)
    iq_mat = 20*np.log10(abs(iq_mat))
    iq_mat = iq_mat-iq_mat.max()

    rs_fname = rel_path / 'Tracks' / ('PALA_InVivoRatBrain_Tracks'+str(dat_num).zfill(3)+'.mat')
    rs_mat = scipy.io.loadmat(rs_fname)
    rs_pts = np.vstack(rs_mat['Track_raw'][0, 0][:, 0])
    del rs_mat

    mat2dict = lambda mat: dict([(k[0], v.squeeze()) for v, k in zip(mat[0][0], list(mat.dtype.descr))])

    P = mat2dict(seq_mat['P'])
    #PData = mat2dict(seq_mat['PData'])
    Trans = mat2dict(seq_mat['Trans'])
    Media = mat2dict(seq_mat['Media'])
    UF = mat2dict(seq_mat['UF'])
    Resource = mat2dict(seq_mat['Resource'])
    TW = mat2dict(seq_mat['TW'])
    TX = mat2dict(seq_mat['TX'])
    Receive = mat2dict(seq_mat['Receive'])
    del seq_mat

    RFdata = rf_mat['RData']
    #Media = mat2dict(rf_mat['Media'])
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

    param.angles_list = P['Angles'] # list of angles [rad] (in TX.Steer)
    param.fnumber = 1.9 # fnumber
    param.compound = 1 # flag to compound [1/0]

    # pixel grid (extracted from PData), in [m]
    origin = np.array([-72,   0,  16])
    size = np.array([ 84, 143,   1])
    pdelta = np.array([1, 0, 1])
    param.x = (origin[0]+np.arange(size[1])*pdelta[2])*param.wavelength
    param.z = (origin[2]+np.arange(size[0])*pdelta[0])*param.wavelength
    [mesh_x, mesh_z] = np.meshgrid(param.x, param.z)

    extent = [min(param.x), max(param.x), min(param.z), max(param.z)]
    aspect = max(param.z)/(max(param.x)-min(param.x))

    cfg.frame_num = 800 if cfg.frame_num is None else cfg.frame_num
    cfg['fs'] = float(Receive['demodFrequency']*1e6)

    # initialize clustering method
    ms = MeanShift(bandwidth=param.wavelength, bin_seeding=True, cluster_all=True, n_jobs=None, max_iter=300, min_bin_freq=1)

    # virtual source (non-planar wave assumption)
    beta = 1e-8
    width = param.xe[-1]-param.xe[0]    # extent of the phased-array
    vsource = [-width*np.cos(param.angles_list[cfg.wave_idx]) * np.sin(param.angles_list[cfg.wave_idx])/beta, -width*np.cos(param.angles_list[cfg.wave_idx])**2/beta]
    nonplanar_tdx = np.hypot((abs(vsource[0])-width/2)*(abs(vsource[0])>width/2), vsource[1])
    src_vec = np.array([vsource[0], vsource[1]])

    # unfold  dimensions
    RFdata = RFdata.reshape(-1, 800, 128, order='F').swapaxes(-2, -1)

    batch_rf_iq_frames = bf_demod_100_bw2iq(RFdata.reshape(3, 640, 128, 800))

    #batch_rf_iq_frames = 20*np.log10(batch_rf_iq_frames)
    #cfg.echo_threshold = float(20*np.log10(cfg.echo_threshold))

    # SVD filter only central plane wave
    if cfg.temp_filter_opt: 
        but_b,but_a = signal.butter(2, np.array([50, 250])/(1000/2), btype='bandpass')
        if cfg.plt_comp_opt or cfg.plt_cluster_opt:
            for wave_idx in range(len(param.angles_list)):
                batch_rf_iq_frames[wave_idx, ...] = svd_filter(batch_rf_iq_frames[wave_idx, ...], cutoff=4)
                batch_rf_iq_frames[wave_idx, ...] = signal.filtfilt(but_b, but_a, batch_rf_iq_frames[wave_idx, ...], axis=2)
        else:
            batch_rf_iq_frames[cfg.wave_idx, ...] = svd_filter(batch_rf_iq_frames[cfg.wave_idx, ...], cutoff=4)
            batch_rf_iq_frames[cfg.wave_idx, ...] = signal.filtfilt(but_b, but_a, batch_rf_iq_frames[cfg.wave_idx, ...], axis=2)

    frame_start = 0
    for frame_batch_ptr in range(frame_start, frame_start+cfg.frame_num, frame_batch_size):

        # rf_iq_frame dimensions: frames x angles x samples x channels
        #rf_iq_frames = np.array([decompose_frame(RFdata[..., frame_idx], len(param.angles_list), int(P['NDsample'])) for frame_idx in range(frame_batch_ptr, frame_batch_ptr+frame_batch_size)])
        rf_iq_frames = batch_rf_iq_frames[..., frame_batch_ptr:frame_batch_ptr+frame_batch_size].swapaxes(-1, -2).swapaxes(-2, -3).swapaxes(-3, 0)

        # blind_zone
        rf_iq_frames[..., :blind_zone_idx, :] = 0

        # crop left and right side of image (transducer array)
        rf_iq_frames_gap = rf_iq_frames#[..., 32:-32]
        param.xe_gap = param.xe#[32:-32]

        rf_iq_frames_gap = rf_iq_frames_gap[:, cfg.wave_idx, :, ::cfg.ch_gap]

        # convert IQ to RF data
        start = time.perf_counter()
        data_batch = iq2rf(np.hstack(rf_iq_frames_gap), mod_freq=param.f0, upsample_factor=cfg.enlarge_factor)
        print('Interpolation time: %s' % str(time.perf_counter()-start))
        
        # BP-filter
        start = time.perf_counter()
        data_batch = bandpass_filter(data_batch, freq_cen=param.f0, freq_smp=param.fs*cfg.enlarge_factor, sw=0.5)
        print('BP-filter time: %s' % str(time.perf_counter()-start))

        # prepare variables for optimization
        data_batch = torch.from_numpy(data_batch.copy()).to(device=cfg.device)
        t = torch.arange(0, len(data_batch[:, 0])/param.fs/cfg.enlarge_factor, 1/param.fs/cfg.enlarge_factor, device=data_batch.device, dtype=data_batch.dtype)

        # power law compensation
        if cfg.pow_law_opt:
            max_val = data_batch.max() * 1.5
            data_batch = compensate_pow_law(data_batch, x=t, a=140.1771, b=1.1578, c=param.c, fkHz=param.f0, sample_rate=param.fs*cfg.enlarge_factor)
            data_batch = data_batch/data_batch.max() * max_val

        # prepare MEMGO performance measurement
        torch.cuda.synchronize()
        start = time.perf_counter()

        # MEMGO optimization
        try:
            memgo_batch, result, conf_batch, echo_batch = batch_staged_memgo(data_batch.T, x=t, max_iter_per_stage=cfg.max_iter, echo_threshold=cfg.echo_threshold, grad_step=cfg.enlarge_factor/6*5, upsample_factor=cfg.enlarge_factor, fs=cfg.fs, print_opt=True)
        except torch._C._LinAlgError:
            continue
        print('MEMGO frame time: %s' % str((time.perf_counter()-start)/frame_batch_size))
        conf_frame = float(torch.nanmean(conf_batch[conf_batch>0]*1e5))
        print('MEMGO confidence: %s' % str(conf_frame))

        if echo_batch.numel() == 0:
            warnings.warn('No echoes found: consider lowering the threshold.')
            continue

        if cfg.logging:
            wandb.log({
                    'PULM/MeanFrameConfidence': conf_frame,
                    'frame': int(frame_batch_ptr+frame_batch_size)*dat_num,
                })
        
        # reshape to dedicated batch dimension
        memgo_batch = memgo_batch.reshape([frame_batch_size, -1, memgo_batch.shape[1], memgo_batch.shape[2]])
        data_batch = data_batch.reshape(len(t), frame_batch_size, -1).swapaxes(0, 1)
        conf_batch = conf_batch.reshape([frame_batch_size, -1, conf_batch.shape[1]])
        echo_batch = echo_batch.reshape([frame_batch_size, -1, echo_batch.shape[1], echo_batch.shape[2]])

        # transfer to data to cpu
        data_batch = data_batch.cpu().numpy()
        conf_batch = conf_batch.cpu().numpy()
        echo_batch = echo_batch.cpu().numpy() if not isinstance(echo_batch, np.ndarray) else echo_batch
        memgo_batch = memgo_batch.cpu().numpy()
        result = result.cpu().numpy()
        t = t.cpu().numpy()

        for frame_batch_idx in range(0, frame_batch_size):

            frame_idx = frame_batch_ptr + frame_batch_idx

            # get raw points from PALA radial symmetry (for validation purposes)
            rs_pts_frame = rs_pts[rs_pts[:, 2] == frame_idx+1]

            print('Dataset: %s, Frame: %s' % (str(dat_num), str(frame_idx)))

            memgo_feats = memgo_batch[frame_batch_idx, ...]
            data_arr = data_batch[frame_batch_idx, ...]
            conf = conf_batch[frame_batch_idx, ...]
            echo_list = echo_batch[frame_batch_idx, ...]

            # beamforming
            if cfg.plt_comp_opt or cfg.plt_cluster_opt:
                #iq_frame = iq_mat['IQ'][..., frame_idx]
                start = time.perf_counter()
                bmode = pala_beamformer(rf_iq_frames[frame_batch_idx, ...], param, mesh_x, mesh_z)
                print('Beamforming time: %s' % str(time.perf_counter()-start))
                bmode -= bmode.max()
                bmode_limits = [-50, 0] # [dB] scale

            # prevent echoes from being in negative time domain
            memgo_feats[memgo_feats[..., 1]<0, 1] = 0

            start = time.perf_counter()
            # select channel
            all_pts_list = []
            rej_pts_list = []
            acc_pts_list = []
            feats = []
            # iterate over transducer gaps (different baselines in parallax terms)
            for tx_gap in cfg.tx_gaps:

                # iterate over channels
                ch_idcs = np.hstack([np.arange(tx_gap+s, len(memgo_feats)-tx_gap, tx_gap) for s in range(tx_gap)])

                # take only non-zero echo locations
                idcs_cch = (memgo_feats[ch_idcs, ..., 1] > 0)

                # take only most confident components
                comp_num = min(conf.shape[1], cfg.comp_max)
                idcs_valid = np.argsort(conf[ch_idcs, :], axis=-1)[:, -comp_num:]

                comps_cch = memgo_feats[ch_idcs[:, None], idcs_valid]
                confs_cch = np.repeat(conf[ch_idcs[:, None], idcs_valid], 3, axis=-1)

                # get central transducers ToA
                cch_dif = t[echo_list[ch_idcs[:, None], idcs_valid, 1].astype(int)] - comps_cch[..., 1]
                toas_cch = (comps_cch[..., 1]+cch_dif+param.t0) * param.c + nonplanar_tdx
                cch_sample, cch_grad = get_amp_grad(data_arr, toas_cch, np.zeros(toas_cch.shape), ch_idcs[:, None])
                toas_cch -= cfg.shift_factor
                cch_cens = (np.array([param.xe_gap[ch_idcs*cfg.ch_gap]*cfg.num_scale, np.zeros(ch_idcs.shape)]) + src_vec[:, None]*cfg.num_scale)/2

                spacer_vec = src_vec[:, None]*cfg.num_scale - np.array([param.xe_gap[ch_idcs*cfg.ch_gap]*cfg.num_scale, np.zeros(ch_idcs.shape)])
                cch_vals = np.array([toas_cch*cfg.num_scale/2, ((toas_cch*cfg.num_scale)**2 - (spacer_vec**2).sum(0)[:, None])**.5 / 2])
                cch_vecs = spacer_vec / (spacer_vec**2).sum(0)**.5

                # get components from adjacent channels
                lch_cidx = np.argmin(abs(np.repeat(comps_cch[..., 1][..., None], memgo_feats.shape[1], axis=-1) - memgo_feats[ch_idcs-tx_gap, :, 1][:, None]), axis=-1)
                rch_cidx = np.argmin(abs(np.repeat(comps_cch[..., 1][..., None], memgo_feats.shape[1], axis=-1) - memgo_feats[ch_idcs+tx_gap, :, 1][:, None]), axis=-1)

                echo_per_sch = 3
                lch_idcs = (np.repeat(lch_cidx[..., None], echo_per_sch, axis=-1) + np.linspace(-1, 1, echo_per_sch).astype('int')).reshape(lch_cidx.shape[0], -1)
                rch_idcs = (np.repeat(rch_cidx[..., None], echo_per_sch, axis=-1) + np.linspace(-1, 1, echo_per_sch).astype('int')).reshape(rch_cidx.shape[0], -1)
                
                # find component index outliers (<0 or >num)
                # tbd = -1
                lch_idcs[(0>lch_idcs) | (lch_idcs>memgo_feats.shape[1]-1)] = 0
                rch_idcs[(0>rch_idcs) | (rch_idcs>memgo_feats.shape[1]-1)] = 0

                # prepare direct neighbour combinations
                comps_lch = memgo_feats[(ch_idcs-tx_gap)[:, None], lch_idcs]
                comps_rch = memgo_feats[(ch_idcs+tx_gap)[:, None], rch_idcs]

                # confidences
                confs_lch = conf[(ch_idcs-tx_gap)[:, None], lch_idcs]
                confs_rch = conf[(ch_idcs+tx_gap)[:, None], rch_idcs]

                # displacements wrt. originally detected echoes
                lch_difs = t[echo_list[(ch_idcs-tx_gap)[:, None], lch_idcs, 1].astype(int)] - comps_lch[..., 1]
                rch_difs = t[echo_list[(ch_idcs+tx_gap)[:, None], rch_idcs, 1].astype(int)] - comps_rch[..., 1]

                # convert overall time-of-arrival distance
                toas_lch = (comps_lch[..., 1]+lch_difs+param.t0) * param.c + nonplanar_tdx
                toas_rch = (comps_rch[..., 1]+rch_difs+param.t0) * param.c + nonplanar_tdx

                # relative phase displacement
                phi_shift_cch = 0
                phi_shifts_lch = (comps_lch[..., 5] - np.repeat(comps_cch[..., 5], echo_per_sch, axis=1))
                phi_shifts_rch = (comps_rch[..., 5] - np.repeat(comps_cch[..., 5], echo_per_sch, axis=1))
                phase_grad_step = 1 #int((param.wavelength/8)/param.c*param.fs*cfg.enlarge_factor)
                phi_shifts_lch = get_overall_phase_shift(data_arr, toas_lch, phi_shifts_lch, (ch_idcs-tx_gap)[:, None], np.repeat(cch_sample, 3, axis=1), np.repeat(cch_grad, 3, axis=1), grad_step=phase_grad_step)
                phi_shifts_rch = get_overall_phase_shift(data_arr, toas_rch, phi_shifts_rch, (ch_idcs+tx_gap)[:, None], np.repeat(cch_sample, 3, axis=1), np.repeat(cch_grad, 3, axis=1), grad_step=phase_grad_step)
                toas_lch -= phi_shifts_lch/(2*np.pi*param.fs) * param.c
                toas_rch -= phi_shifts_rch/(2*np.pi*param.fs) * param.c

                toas_lch -= cfg.shift_factor
                toas_rch -= cfg.shift_factor

                # set ellipse parameters
                xe_positions = np.array([[param.xe_gap[(ch_idcs-tx_gap)*cfg.ch_gap], np.zeros(ch_idcs.shape)], [param.xe_gap[(ch_idcs+tx_gap)*cfg.ch_gap], np.zeros(ch_idcs.shape)]])
                spacer_vecs = (src_vec[None, :, None] - xe_positions) * cfg.num_scale
                sch_cens = (xe_positions + src_vec[None, :, None]) / 2 * cfg.num_scale
                sch_vecs = spacer_vecs / (spacer_vecs**2).sum(1)**.5
                sch_vecs = np.dstack(sch_vecs).reshape(2, -1)
                sch_vals = np.array([np.array([toas_lch, toas_rch])*cfg.num_scale/2, ((np.array([toas_lch, toas_rch])*cfg.num_scale)**2 - (spacer_vecs**2).sum(1)[..., None])**.5 / 2])
                echo_cch_num = idcs_valid.shape[1]
                ch_num = len(ch_idcs)
                # move channel dimension to the front
                cch_cens = np.swapaxes(cch_cens, 0, 1)
                sch_cens = np.swapaxes(np.swapaxes(sch_cens,1,2), 0, 1)
                cch_vals = np.swapaxes(cch_vals, 0, 1)
                cch_vecs = np.swapaxes(cch_vecs, 0, 1)

                cen_cens = np.stack([np.repeat(cch_cens[:, 0], 2*echo_per_sch*echo_cch_num), np.repeat(cch_cens[:, 1], 2*echo_per_sch*echo_cch_num)])
                adj_cens = np.stack([np.ascontiguousarray(np.repeat(sch_cens[..., 0], echo_per_sch*echo_cch_num)), np.ascontiguousarray(np.repeat(sch_cens[..., 1], echo_per_sch*echo_cch_num))])
                cen_vals = np.stack([np.repeat(np.repeat(cch_vals[:, 0, :], echo_per_sch, axis=1), 2, axis=0).flatten(), np.repeat(np.repeat(cch_vals[:, 1, :], echo_per_sch, axis=1), 2, axis=0).flatten()])
                adj_vals = np.stack([np.ascontiguousarray(np.swapaxes(sch_vals[0],0,1).flatten()), np.ascontiguousarray(np.swapaxes(sch_vals[1],0,1).flatten())])
                cen_vecs = np.stack([np.repeat(cch_vecs[:, 0], 2*echo_per_sch*echo_cch_num), np.repeat(cch_vecs[:, 1], 2*echo_per_sch*echo_cch_num)])
                adj_vecs = np.stack([np.ascontiguousarray(np.repeat(sch_vecs[0], echo_per_sch*echo_cch_num)), np.ascontiguousarray(np.repeat(sch_vecs[1], echo_per_sch*echo_cch_num))])
                
                # pass ellipse properties
                ell_intersector.set_all_ellipses(
                    cen_cens[0], cen_cens[1],
                    adj_cens[0], adj_cens[1],
                    cen_vals[0], cen_vals[1],
                    adj_vals[0], adj_vals[1],
                    cen_vecs[0], cen_vecs[1],
                    adj_vecs[0], adj_vecs[1],
                )

                spts, valid_intersects = ell_intersector.get_intersection_multiple()
                spts /= cfg.num_scale
                pts_mask = (min(param.x) < spts[..., 0]) & (spts[..., 0] < max(param.x)) & (min(param.z) < spts[..., 1]) & (spts[..., 1] < max(param.z))
                pts = spts[pts_mask]
                pts_mask_num = np.any(pts_mask > 0, axis=1)
                pts_mask_num &= valid_intersects

                pts_idcs = ~np.repeat(np.dstack([np.ones_like(ch_idcs), np.zeros_like(ch_idcs)]).flatten(), echo_per_sch*comp_num).astype(bool)[pts_mask_num]

                if cfg.parity_opt:
                    # parity toa distance of neighbor transducer
                    par_ch_idcs = np.repeat(np.dstack([ch_idcs+tx_gap, ch_idcs-tx_gap]).flatten(), echo_per_sch*comp_num)[pts_mask_num]
                    
                    tx_pos = np.vstack([param.xe_gap[par_ch_idcs*cfg.ch_gap], np.zeros(par_ch_idcs.shape[0])]).T
                    virtual_tdx = np.hypot(pts[:, 0]-vsource[0], pts[:, 1]-vsource[1])
                    dtx = virtual_tdx - nonplanar_tdx
                    mu_pars = (np.linalg.norm(pts-tx_pos, axis=-1)+dtx) / param.c - param.t0

                    idx_pars = np.argmin(abs(mu_pars[:, None] - memgo_feats[par_ch_idcs, :, 1]), axis=-1)
                    comp_pars = memgo_feats[par_ch_idcs, idx_pars, :]

                    echo_pars = t[echo_list[par_ch_idcs, idx_pars, 1].astype(int)]
                    par_difs = echo_pars - comp_pars[:, 1]
                    toa_pars = (comp_pars[:, 1]+par_difs+param.t0) * param.c + nonplanar_tdx
                    
                    # comp_cch indices to which each comp_par belongs to
                    cch_idx_pars = np.concatenate([np.repeat(np.arange(echo_cch_num), echo_per_sch), np.repeat(np.arange(echo_cch_num), echo_per_sch)])
                    cch_idx_pars = np.repeat(cch_idx_pars[None, :], len(ch_idcs), axis=0)
                    s = np.array([comp_cch[cch_idx_par] for (comp_cch, cch_idx_par) in zip(comps_cch, cch_idx_pars)]).reshape(-1, 6)

                    phi_shift_pars = comp_pars[:, 5] - s[pts_mask_num, 5]
                    cch_sample_par = np.repeat(np.concatenate([cch_sample, cch_sample], axis=-1).flatten(), echo_per_sch)[pts_mask_num]
                    cch_grad_par = np.repeat(np.concatenate([cch_grad, cch_grad], axis=-1).flatten(), echo_per_sch)[pts_mask_num]
                    phi_shift_pars = get_overall_phase_shift(data_arr, toa_pars, phi_shift_pars, par_ch_idcs, cch_sample_par, cch_grad_par)

                    toa_pars -= phi_shift_pars/(2*np.pi*param.fs) * param.c
                    dist_pars = abs((toa_pars-nonplanar_tdx)/param.c-param.t0 - mu_pars) * param.fs

                    valid = dist_pars < cfg.dist_par_threshold
                else:
                    dist_pars = np.ones(pts.shape[0])*float('NaN')
                    valid = np.ones(pts.shape[0], dtype=bool)

                if pts.size > 0: all_pts_list.append(np.array([pts[valid, 0], pts[valid, 1]]).T)
                if pts.size > 0: rej_pts_list.append(np.array([pts[~valid, 0], pts[~valid, 1]]).T)

                if cfg.plt_comp_opt:
                    
                    cch_idcs_flat = np.repeat(np.repeat(np.repeat(ch_idcs, comp_num).reshape(-1, comp_num), echo_per_sch, axis=1), 2, axis=0).flatten()[pts_mask_num]
                    sch_idcs_flat = np.repeat(np.dstack([ch_idcs-tx_gap, ch_idcs+tx_gap]).flatten(), echo_per_sch*comp_num)[pts_mask_num]
                    
                    cor_ch_idcs = np.repeat(np.dstack([ch_idcs-tx_gap, ch_idcs+tx_gap]).flatten(), echo_per_sch*comp_num)[pts_mask_num]
                    nidx_pars = np.argmin(abs(mu_pars[:, None] - memgo_feats[cor_ch_idcs, :, 1]), axis=-1)
                    sch_comps = memgo_feats[sch_idcs_flat, nidx_pars, :]
                    sch_phi_shifts = np.dstack([phi_shifts_lch, phi_shifts_rch]).swapaxes(1, -1).flatten()[pts_mask_num]
                    sch_phi_shifts = np.array([phi_shifts_lch, phi_shifts_rch]).flatten()[pts_mask_num]
                    sch_comps = np.array([comps_lch[..., 1], comps_rch[..., 1]]).flatten()[pts_mask_num]

                    for k, pt in enumerate(pts):
                        
                        plt.rcParams.update({'font.size': 18})
                        fig = plt.figure(figsize=(30/3*1.4, 15/3))
                        gs = gridspec.GridSpec(2, 2)
                        ax1 = plt.subplot(gs[:, 1])
                        ax2 = plt.subplot(gs[1, 0])
                        ax3 = plt.subplot(gs[0, 0])

                        ax1.imshow(bmode, vmin=bmode_limits[0], vmax=bmode_limits[1], extent=extent, aspect=aspect**-1, cmap='gray', origin='lower')
                        ax1.set_facecolor('#000000')
                        ax1.plot([min(param.xe_gap), max(param.xe_gap)], [0, 0], color='gray', linewidth=5, label='Transducer plane')
                        xzc = np.array([cen_cens[:, pts_mask_num][:, k][0], cen_cens[:, pts_mask_num][:, k][1]]) / cfg.num_scale
                        ax1.set_ylim([0, max(param.z)])#ax1.set_ylim([min(xzc), max(abs(xzc))])#
                        ax1.set_xlim([min(param.x), max(param.x)])#ax1.set_xlim([min(xzc), max(abs(xzc))])#
                        #ax1.set_xlabel('Horizontal domain $x$ [m]')
                        #ax1.set_ylabel('Vertical domain $z$ [m]')
                        #ax1.yaxis.set_label_position("right")
                        #ax1.yaxis.tick_right()

                        ax1.plot(pt[0], pt[1], '1', color='cyan', markersize=8+2, label='Intersections')
                        #ax1.text(pt[0], pt[1], s=str(dist_pars[k]), color='w')
                        ax1.legend()

                        mu_cch = np.repeat(np.repeat(comps_cch[..., 1], echo_per_sch, axis=1), 2, axis=0).flatten()[pts_mask_num][k] * param.fs * cfg.enlarge_factor
                        
                        # when is index k for left channel and when for right? answer: pts_idcs
                        pts_idx = int(pts_idcs[k])
                        for j, (ax, cen, val, vec, color) in enumerate(zip([ax2, ax3], [cen_cens[:, pts_mask_num][:, k], adj_cens[:, pts_mask_num][:, k]], [cen_vals[:, pts_mask_num][:, k], adj_vals[:, pts_mask_num][:, k]], [cen_vecs[:, pts_mask_num][:, k], adj_vecs[:, pts_mask_num][:, k]], ['g']+[['royalblue', 'y'][pts_idx]])):
                            
                            el_idx = cch_idcs_flat[k] if j==0 else sch_idcs_flat[k]
                            dmax = max([max(data_arr[:, cch_idcs_flat[k]]), max(data_arr[:, sch_idcs_flat[k]])])
                            dmin = min([min(data_arr[:, cch_idcs_flat[k]]), min(data_arr[:, sch_idcs_flat[k]])])
                                
                            # ellipse plot
                            #val -= cfg.shift_factor*cfg.num_scale/2
                            major_axis_radius = np.linalg.norm(vec*val[0] / cfg.num_scale)
                            minor_axis_radius = np.linalg.norm(vec*val[1] / cfg.num_scale)
                            xz = np.array([cen[0], cen[1]]) / cfg.num_scale
                            # use float128 as small angles yield zero otherwise
                            vector_a = np.longdouble([(param.xe_gap[el_idx*cfg.ch_gap] - vsource[0]), vsource[1]])
                            vector_b = np.longdouble([0, -1])#np.longdouble([vsource[0], vsource[1]])
                            angle_deg = np.arccos(np.dot(vector_a, vector_b) / (np.linalg.norm(vector_a) * np.linalg.norm(vector_b))) / np.pi * 180
                            angle_deg *= np.sign(vsource[0]-cen[0]) * np.sign(vsource[0]+np.spacing(1)) #-1 *np.sign(param.xe_gap[el_idx*cfg.ch_gap])
                            ell = Ellipse(xy=xz, width=2*minor_axis_radius, height=2*major_axis_radius, angle=angle_deg, edgecolor=color, linewidth=3, fc='None', rasterized=True)
                            ax1.add_artist(ell)

                            # plot detected mu echoes param
                            mus_samples = np.stack([(memgo_feats[el_idx, :, 1]) * param.fs * cfg.enlarge_factor,]*2)
                            mus_ys = np.stack([np.array([dmin, dmax]),]*memgo_feats.shape[-2]).T
                            ax.plot(mus_samples, mus_ys, color='gray')
                            ax.plot([[e[1] for e in echo_list[el_idx]], [e[1] for e in echo_list[el_idx]]], mus_ys[:, :len(echo_list[el_idx])], color='k')

                            # plot data and fitted result
                            ax.plot(np.abs(signal.hilbert(data_arr[:, el_idx])), label='Envelope', color='gray', alpha=.4)
                            ax.plot(data_arr[:, el_idx], label='Raw signal', color=color)
                            ax.plot(result[el_idx, ...], label='Reconstructed', color='black', linestyle='dashed')

                            ax.set_xlabel('Radial distance $r$ [samples]')
                            #ax.set_ylabel('Amplitude $A_{%s}(r)$ [a.u.]' % el_idx)
                            ax.set_ylabel('Amplitude [a.u.]')
                            #ax.set_xlim([mu_cch-500, mu_cch+500])
                            ax.grid(True)

                            # plot rx trajectory
                            ax1.plot([param.xe_gap[el_idx*cfg.ch_gap], pt[0]], [0, pt[1]], color, linewidth=3, linestyle='dashed', label='Rx path ch. %s' % el_idx)
                            ax1.legend(loc='lower right')
                            
                        # plot components
                        ax2.plot(np.stack([mu_cch,]*2), [dmin, dmax], color='red', label='Time-of-Arrival')
                        ax3.plot(np.stack([(sch_comps[k] - sch_phi_shifts[k]/(2*np.pi*param.fs)) * param.fs * cfg.enlarge_factor,]*2), [dmin, dmax], color='red', label='Time-of-Arrival')
                        ax2.legend(loc='lower left')
                        ax3.legend(loc='lower left')
                        ax2.set_title('Ch. 1')
                        ax3.set_title('Ch. 0')
                        
                        # switch all ticks off
                        ax1.tick_params(top=False, bottom=False, left=False, right=False, labelleft=False, labelbottom=False)
                        ax2.tick_params(top=False, bottom=False, left=False, right=False, labelleft=False, labelbottom=False)
                        ax3.tick_params(top=False, bottom=False, left=False, right=False, labelleft=False, labelbottom=False)

                        # switch off ticks and labels for shared axis
                        plt.setp(ax3.get_xticklabels(), visible=False)
                        ax3.set_xlabel(xlabel='', visible=False)

                        #ax3.plot(np.stack([((toa_pars[k]-nonplanar_tdx)/param.c-param.t0) * param.fs * cfg.enlarge_factor,]*2), [min(result[par_ch_idcs[k], :]), max(result[par_ch_idcs[k], :])], color='pink', linestyle='dashdot', linewidth=2)
                        plt.tight_layout()#pad=1.8)
                        if k == 10:
                            fig.patch.set_alpha(0)  # transparency
                            plt.savefig('./components_plot.pdf', format='pdf', backend='pdf', dpi=300, transparent=False)
                            print('saved')
                        plt.show()
            
            if len(all_pts_list) == 0:
                np.savetxt((output_path / ('pace_frame_%s_%s.csv' % (str(dat_num).zfill(3), str(frame_idx).zfill(4)))), np.array([]), delimiter=',')
                continue
            else:
                all_pts = np.vstack(all_pts_list)

            if len(rej_pts_list):
                rej_pts = np.vstack(rej_pts_list)

            if all_pts.size > 0:
                ms.fit(all_pts[:, :2])
                labels = ms.labels_
                cluster_centers = ms.cluster_centers_
            else:
                np.savetxt((output_path / ('pace_frame_%s_%s.csv' % (str(dat_num).zfill(3), str(frame_idx).zfill(4)))), np.array([]), delimiter=',')
                continue

            labels_unique = np.unique(labels)
            n_clusters_ = len(labels_unique)

            print("number of estimated clusters : %d" % n_clusters_)

            reduced_pts = []
            for i, l in enumerate(labels_unique):
                if l == -1:
                    continue
                # select point closest to the mean
                label_xy_mean = np.mean(all_pts[l==labels], axis=0)[:2]
                idx = np.argmin(np.sum((all_pts[l==labels][:, :2] - label_xy_mean)**2, axis=-1))

                if sum(l==labels) > cfg.cluster_number: reduced_pts.append(label_xy_mean)#all_pts[l==labels][idx]) #

            print('Frame time: %s' % str(time.perf_counter()-start))

            pace_arr = np.array(reduced_pts)[:, :2] if np.array(reduced_pts).size > 0 else np.array([])

            if cfg.save_opt: np.savetxt((output_path / ('pace_frame_%s_%s.csv' % (str(dat_num).zfill(3), str(frame_idx).zfill(4)))), pace_arr, delimiter=',')

            if cfg.plt_cluster_opt:
                plt.rcParams.update({'font.size': 18})
                fig = plt.figure(figsize=(30/3*1.45, 15/3))
                gs = gridspec.GridSpec(1, 2)
                ax1 = plt.subplot(gs[0, 0])
                ax2 = plt.subplot(gs[0, 1])

                ax1.imshow(bmode, vmin=bmode_limits[0], vmax=bmode_limits[1], extent=extent, aspect=aspect**-1, interpolation='none', origin='lower', cmap='gray')
                ax1.set_facecolor('#000000')
                ax1.plot([min(param.xe_gap), max(param.xe_gap)], [0, 0], color='gray', linewidth=5, label='Transducer plane')
                #ax1.plot((rs_pts_frame[:, 1]-59)*param.wavelength, rs_pts_frame[:, 0]*param.wavelength, 'b+', label='RS', alpha=1)
                ax1.plot((rs_pts_frame[:, 1]-63)*param.wavelength, (rs_pts_frame[:, 0]+25)*param.wavelength, 'b+', label='RS', alpha=1)
                ax1.plot(all_pts[:, 0], all_pts[:, 1], 'gx', label='all points', alpha=.4)
                ax1.plot(rej_pts[:, 0], rej_pts[:, 1], '.', color='orange', label='rejected points', alpha=.2)
                #[ax1.text(rej_pts[i, 0], rej_pts[i, 1]+np.random.rand(1)*param.wavelength, s=str(rej_pts[i, 2]), color='orange') for i in range(len(rej_pts))]
                if len(reduced_pts)>0: ax1.plot(np.array(reduced_pts)[:, 0], np.array(reduced_pts)[:, 1], 'c+', label='selected')
                ax1.set_ylim([0, max(param.z)])
                denom = 1.2
                ax1.set_xlim([min(param.x)/denom, max(param.x)/denom])
                #ax1.set_xlabel('Horizontal domain $x$ [m]')
                #ax1.set_ylabel('Vertical distance $z$ [m]')
                ax1.legend()

                ax2.plot(rs_pts_frame[:, 1], rs_pts_frame[:, 0], 'b+', label='RS', alpha=1)
                ax2.imshow(iq_mat[..., frame_idx], vmin=bmode_limits[0]-20, vmax=bmode_limits[1]-20, interpolation='none', origin='lower', cmap='gray')
                #if len(reduced_pts)>0: ax2.plot(np.array(reduced_pts)[:, 0]/param.wavelength+59, np.array(reduced_pts)[:, 1]/param.wavelength, 'c+', label='selected')
                if len(reduced_pts)>0: ax2.plot(np.array(reduced_pts)[:, 0]/param.wavelength+63, np.array(reduced_pts)[:, 1]/param.wavelength-25, 'c+', label='selected')

                #for i, l in enumerate(labels_unique):
                #    if sum(l==labels) > cfg.cluster_number: 
                #        ax1.plot(all_pts[:, 0][l==labels], all_pts[:, 1][l==labels], marker='.', linestyle='', color=['brown', 'pink', 'yellow', 'white', 'gray', 'violet', 'green', 'blue'][i%8])
                #for i, tx_gap_pts in enumerate(all_pts_list):
                #    ax1.plot(tx_gap_pts[:, 0], tx_gap_pts[:, 1], marker='.', linestyle='', color=['brown', 'pink', 'yellow', 'white', 'gray', 'cyan', 'green', 'blue'][i%8], label=str(cfg.tx_gaps[i]))
                ax1.legend()
                plt.savefig('./cluster_plot.pdf', format='pdf', backend='pdf', dpi=300, transparent=True)
                plt.show()

    if cfg.save_opt:
        frames = load_ulm_data(data_path=str(output_path), expr='pace')
        pace_ulm_img, _ = render_ulm(frames, tracking=cfg.tracking, plot_opt=cfg.plt_frame_opt, cmap_opt=True, uint8_opt=False, gamma=cfg.gamma, srgb_opt=True)
        if cfg.logging:
            wandb.log({"pace_ulm_img": wandb.Image(pace_ulm_img)}, step=dat_num)
