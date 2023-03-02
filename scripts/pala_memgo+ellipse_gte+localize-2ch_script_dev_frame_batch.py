import sys
sys.path.append('../')

import scipy.io
from scipy import signal
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Ellipse
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes, mark_inset
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
import torch
import time
from sklearn.cluster import MeanShift, estimate_bandwidth
from pathlib import Path
from omegaconf import OmegaConf
import wandb
from skimage.metrics import structural_similarity
from scipy.interpolate import interp1d

from multimodal_emg.batch_staged_memgo import batch_staged_memgo
from gte_intersect.ellipse import EllipseIntersection
from utils.pala_beamformer import pala_beamformer, decompose_frame
from utils.pala_error import rmse_unique
from utils.render_ulm import render_ulm, load_ulm_data
from utils.iq2rf import iq2rf
from utils.speckle_noise import add_pala_noise
from utils.bandpass import bandpass_filter


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
cfg = OmegaConf.load(str(script_path.parent / 'config.yaml'))

# override config with CLI
cfg = OmegaConf.merge(cfg, OmegaConf.from_cli())

assert (1000/cfg.frame_batch_size)%1 == 0, 'frame_batch_size must be multiple of 1000'

if cfg.logging: 
    wandb.init(project="pulm", name=None, config=cfg, group=None)
    wandb.define_metric('PULM/RMSE', step_metric='frame')
    wandb.define_metric('PULM/Precision', step_metric='frame')
    wandb.define_metric('PULM/Recall', step_metric='frame')
    wandb.define_metric('PULM/Jaccard', step_metric='frame')
    wandb.define_metric('PULM/MeanFrameConfidence', step_metric='frame')
    wandb.define_metric('PULM/TruePositive', step_metric='frame')
    wandb.define_metric('PULM/FalsePositive', step_metric='frame')
    wandb.define_metric('PULM/FalseNegative', step_metric='frame')
    wandb.define_metric('PALA/RMSE', step_metric='frame')
    wandb.define_metric('PALA/Precision', step_metric='frame')
    wandb.define_metric('PALA/Recall', step_metric='frame')
    wandb.define_metric('PALA/Jaccard', step_metric='frame')
    wandb.define_metric('PALA/MeanFrameConfidence', step_metric='frame')
    wandb.define_metric('PALA/TruePositive', step_metric='frame')
    wandb.define_metric('PALA/FalsePositive', step_metric='frame')
    wandb.define_metric('PALA/FalseNegative', step_metric='frame')

output_path = script_path / 'other_frames'
if cfg.save_opt and not output_path.exists(): output_path.mkdir()

if cfg.plt_comp_opt or cfg.plt_cluster_opt:
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

acc_pace_errs = []
acc_pala_errs = []
for dat_num in range(1, cfg.dat_num):

    cfg_fname = rel_path / 'PALA_InSilicoFlow_v3_config.mat'
    cfg_mat = scipy.io.loadmat(cfg_fname)

    seq_fname = rel_path / 'PALA_InSilicoFlow_sequence.mat'
    seq_mat = scipy.io.loadmat(seq_fname)

    pos_fname = rel_path / 'PALA_InSilicoFlow_v3_pos_Tracks_dt.mat'
    pos_mat = scipy.io.loadmat(pos_fname)

    if np.isreal(cfg.noise_db) and cfg.noise_db < 0:
        assert cfg.noise_db in [-30, -40, -50], 'Noise level not available for PALA'
        res_fname = rel_path / 'Results' / 'matlab_w_noise' / ('PALA_InSilicoFlow_raw_db-'+str(abs(cfg.noise_db))+'_'+str(dat_num)+'.mat')
    else:
        res_fname = rel_path / 'Results' / 'matlab_wo_noise' / ('PALA_InSilicoFlow_raw_'+str(dat_num)+'.mat')
    track_key = 'Track_raw'
    res_mat = scipy.io.loadmat(res_fname)

    rf_fname = rel_path / 'RF' / ('PALA_InSilicoFlow_RF'+str(dat_num).zfill(3)+'.mat')
    rf_mat = scipy.io.loadmat(rf_fname)

    if cfg.plt_comp_opt or cfg.plt_cluster_opt:
        iq_fname = rel_path / 'IQ' / ('PALA_InSilicoFlow_IQ'+str(dat_num).zfill(3)+'.mat')
        iq_mat = scipy.io.loadmat(iq_fname)

    pala_local_methods = [el[0] for el in res_mat['listAlgo'][0]]
    pala_local_results = {m: arr for arr, m in zip(res_mat[track_key][0], pala_local_methods)}
    pala_method = pala_local_methods[-1]
    ref_pts = pala_local_results[pala_method]
    del res_mat

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
    del seq_mat

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
    aspect = max(param.z)/(max(param.x)-min(param.x))#1

    cfg.frame_num = rf_mat['RFdata'].shape[-1] if cfg.frame_num is None else cfg.frame_num
    cfg['fs'] = float(Receive['demodFrequency']*1e6)

    # initialize clustering method
    ms = MeanShift(bandwidth=param.wavelength, bin_seeding=True, cluster_all=True, n_jobs=None, max_iter=300, min_bin_freq=1)

    # virtual source (non-planar wave assumption)
    beta = 1e-8
    width = param.xe[-1]-param.xe[0]    # extent of the phased-array
    vsource = [-width*np.cos(param.angles_list[cfg.wave_idx]) * np.sin(param.angles_list[cfg.wave_idx])/beta, -width*np.cos(param.angles_list[cfg.wave_idx])**2/beta]
    nonplanar_tdx = np.hypot((abs(vsource[0])-width/2)*(abs(vsource[0])>width/2), vsource[1])
    src_vec = np.array([vsource[0], vsource[1]])

    frame_start = 0
    for frame_batch_ptr in range(frame_start, frame_start+cfg.frame_num, frame_batch_size):

        # rf_iq_frame dimensions: frames x angles x samples x channels
        rf_iq_frames = np.array([decompose_frame(RFdata[..., frame_idx], int(P['numTx']), int(P['NDsample'])) for frame_idx in range(frame_batch_ptr, frame_batch_ptr+frame_batch_size)])

        # convert IQ to RF data
        data_batch = iq2rf(np.hstack(rf_iq_frames[:, cfg.wave_idx, :, ::cfg.ch_gap]), mod_freq=param.f0, upsample_factor=10)

        if np.isreal(cfg.noise_db) and cfg.noise_db < 0:
            # add noise according to PALA study
            data_batch = add_pala_noise(data_batch, clutter_db=cfg.noise_db)
            # bandpass filter to counteract impact of noise
            start = time.perf_counter()
            data_batch = bandpass_filter(data_batch, freq_cen=param.f0, freq_smp=param.fs*10)#cfg.enlarge_factor)
            print('BP-filter time: %s' % str(time.perf_counter()-start))

        # upsample
        start = time.perf_counter()
        x = np.linspace(0, len(data_batch)/param.f0, num=len(data_batch), endpoint=True)
        t = np.linspace(0, len(data_batch)/param.f0, num=int(len(data_batch)*cfg.enlarge_factor/10), endpoint=True)
        f = interp1d(x, data_batch, axis=0)
        data_batch = f(t)
        print('Interpolation time: %s' % str(time.perf_counter()-start))

        # prepare variables for optimization
        data_batch = torch.from_numpy(data_batch.copy()).to(device=cfg.device)
        t = torch.arange(0, len(data_batch[:, 0])/param.fs/cfg.enlarge_factor, 1/param.fs/cfg.enlarge_factor, device=data_batch.device, dtype=data_batch.dtype)

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

            print('Dataset: %s, Frame: %s' % (str(dat_num), str(frame_idx)))

            memgo_feats = memgo_batch[frame_batch_idx, ...]
            data_arr = data_batch[frame_batch_idx, ...]
            conf = conf_batch[frame_batch_idx, ...]
            echo_list = echo_batch[frame_batch_idx, ...]

            # ground truth
            xpos = ListPos[:, 0, frame_idx] * param.wavelength
            zpos = ListPos[:, 2, frame_idx] * param.wavelength

            # PALA
            pts_frame_idcs = ref_pts[:, -1]-1 == frame_idx
            ref_imax = ref_pts[pts_frame_idcs][:, 0]    # intensity max
            ref_zpos = ref_pts[pts_frame_idcs][:, 1]
            ref_xpos = ref_pts[pts_frame_idcs][:, 2]

            # beamforming
            if cfg.plt_comp_opt or cfg.plt_cluster_opt:
                iq_frame = iq_mat['IQ'][..., frame_idx]
                start = time.perf_counter()
                bmode = pala_beamformer(rf_iq_frames[frame_batch_idx, ...], param, mesh_x, mesh_z)
                print('Beamforming time: %s' % str(time.perf_counter()-start))
                bmode -= bmode.max()
                bmode_limits = [-60, 0] # [dB] scale

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
                cch_cens = (np.array([param.xe[ch_idcs*cfg.ch_gap]*cfg.num_scale, np.zeros(ch_idcs.shape)]) + src_vec[:, None]*cfg.num_scale)/2

                spacer_vec = src_vec[:, None]*cfg.num_scale - np.array([param.xe[ch_idcs*cfg.ch_gap]*cfg.num_scale, np.zeros(ch_idcs.shape)])
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
                xe_positions = np.array([[param.xe[(ch_idcs-tx_gap)*cfg.ch_gap], np.zeros(ch_idcs.shape)], [param.xe[(ch_idcs+tx_gap)*cfg.ch_gap], np.zeros(ch_idcs.shape)]])
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
                    
                    tx_pos = np.vstack([param.xe[par_ch_idcs*cfg.ch_gap], np.zeros(par_ch_idcs.shape[0])]).T
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

                    valid = dist_pars < cfg.dist_par_threshold  # .5
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
                        fig = plt.figure(figsize=(30/3*1.3, 15/3))
                        gs = gridspec.GridSpec(2, 2)
                        ax1 = plt.subplot(gs[:, 1])
                        ax2 = plt.subplot(gs[1, 0])
                        ax3 = plt.subplot(gs[0, 0])

                        # select nearest point position
                        gt_pt_idx = np.argmin(abs(np.array([xpos[~np.isnan(xpos)], zpos[~np.isnan(zpos)]]) - pt[:, None]).sum(0))
                        pa_pt_idx = np.argmin(abs(np.array([(ref_xpos)*param.wavelength, (ref_zpos)*param.wavelength]) - pt[:, None]).sum(0))
                        gt_pt = np.array([xpos[~np.isnan(xpos)], zpos[~np.isnan(zpos)]])[:, gt_pt_idx]
                        pa_pt = np.array([(ref_xpos)*param.wavelength, (ref_zpos)*param.wavelength])[:, pa_pt_idx]

                        ax1.imshow(bmode, vmin=bmode_limits[0], vmax=bmode_limits[1], extent=extent, aspect=aspect**-1, cmap='gray', origin='lower')
                        ax1.set_facecolor('#000000')
                        ax1.plot(gt_pt[0], gt_pt[1], 'rx', markersize=12, label='Ground truth')
                        ax1.plot(pa_pt[0], pa_pt[1], 'b+', markersize=12, label='Radial symmetry')
                        ax1.plot(pt[0], pt[1], '1', color='cyan', markersize=12+2, label='Intersection')
                        xzc = np.array([cen_cens[:, pts_mask_num][:, k][0], cen_cens[:, pts_mask_num][:, k][1]]) / cfg.num_scale
                        ax1.set_ylim([0, max(param.z)])#ax1.set_ylim([min(xzc), max(abs(xzc))])#
                        ax1.set_xlim([min(param.x), max(param.x)])#ax1.set_xlim([min(xzc), max(abs(xzc))])#
                        #ax1.set_xlabel('Horizontal domain $x$ [m]')
                        #ax1.set_ylabel('Vertical domain $z$ [m]')

                        mu_cch = np.repeat(np.repeat(comps_cch[..., 1], echo_per_sch, axis=1), 2, axis=0).flatten()[pts_mask_num][k] * param.fs * cfg.enlarge_factor
                        
                        # when is index k for left channel and when for right? answer: pts_idcs
                        pts_idx = int(pts_idcs[k])
                        axins1_ells = []
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
                            vector_a = np.longdouble([(param.xe[el_idx*cfg.ch_gap] - vsource[0]), vsource[1]])
                            vector_b = np.longdouble([0, -1])#np.longdouble([vsource[0], vsource[1]])
                            angle_deg = np.arccos(np.dot(vector_a, vector_b) / (np.linalg.norm(vector_a) * np.linalg.norm(vector_b))) / np.pi * 180
                            angle_deg *= np.sign(vsource[0]-cen[0]) * np.sign(vsource[0]+np.spacing(1)) #-1 *np.sign(param.xe[el_idx*cfg.ch_gap])
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
                            ax.set_xlim([mu_cch-600, mu_cch+400])
                            ax.grid(True)

                            # plot rx trajectory
                            ax1.plot([param.xe[el_idx*cfg.ch_gap], pt[0]], [0, pt[1]], color, linewidth=3, linestyle='dashed', label='Rx path ch. %s' % el_idx)

                            #axins1.plot([param.xe[el_idx*cfg.ch_gap], pt[0]], [0, pt[1]], color, linewidth=3, linestyle='dashed', label='Rx path ch. %s' % el_idx)

                            ell_axins1 = Ellipse(xy=xz, width=2*minor_axis_radius, height=2*major_axis_radius, angle=angle_deg, edgecolor=color, linewidth=3, fc='None', rasterized=True)
                            axins1_ells.append(ell_axins1)
                        
                        # finish frame axis
                        ax1.plot([min(param.x), max(param.x)], [0, 0], color='gray', linewidth=8, label='Transducer plane')
                        ax1.legend(framealpha=1)

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
                        plt.tight_layout()#pad=1.8

                        # plot zoomed frame
                        w = 3.5*param.wavelength
                        axins1 = zoomed_inset_axes(ax1, zoom=6+1, loc='upper right')
                        axins1.imshow(bmode, extent=extent, aspect=aspect**-1, origin='lower', cmap='gray')
                        x1, x2, y1, y2 = gt_pt[0]-w, gt_pt[0]+w, gt_pt[1]-w*aspect, gt_pt[1]+w*aspect
                        axins1.set_xlim(x1, x2)
                        axins1.set_ylim(y1, y2)
                        axins1.yaxis.get_major_locator().set_params(nbins=7)
                        axins1.xaxis.get_major_locator().set_params(nbins=7)
                        axins1.tick_params(top=False, bottom=False, left=False, right=False, labelleft=False, labelbottom=False)
                        mark_inset(ax1, axins1, loc1=2, loc2=3, fc="none", ec='orange', lw=2)
                        
                        axins1.plot(gt_pt[0], gt_pt[1], 'rx', markersize=15, label='Ground truth')
                        axins1.plot(pa_pt[0], pa_pt[1], 'b+', markersize=15, label='Radial symmetry')   #str(pala_method)
                        for ell in axins1_ells:
                            axins1.add_artist(ell)
                        axins1.plot(pt[0], pt[1], '1', color='cyan', markersize=15+2, label='Intersection')
                            
                        # style for frame
                        sides_list = ['bottom', 'top', 'right', 'left']
                        [axins1.spines[s].set_color('orange') for s in sides_list]
                        [axins1.spines[s].set_linewidth(2) for s in sides_list]

                        if k == 10+0:
                            fig.patch.set_alpha(0)  # transparency
                            plt.savefig('./components_plot.pdf', format='pdf', backend='pdf', dpi=300, transparent=False)
                            print('saved')
                        plt.close()
                        #plt.show()

            all_pts = np.vstack(all_pts_list)
            rej_pts = np.vstack(rej_pts_list)

            ms.fit(all_pts[:, :2])
            labels = ms.labels_
            cluster_centers = ms.cluster_centers_

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

            gtru_arr = np.array([xpos, zpos]).T[(~np.isnan(xpos)) & (~np.isnan(zpos)), :]
            pace_arr = np.array(reduced_pts)[:, :2] if np.array(reduced_pts).size > 0 else np.array([])
            pala_arr = np.array([(ref_xpos)*param.wavelength, (ref_zpos)*param.wavelength]).T
            pace_err, pace_p, pace_r, pace_jidx, pace_tp, pace_fp, pace_fn = rmse_unique(pace_arr/param.wavelength, gtru_arr/param.wavelength, tol=1/4)
            pala_err, pala_p, pala_r, pala_jidx, pala_tp, pala_fp, pala_fn = rmse_unique(pala_arr/param.wavelength, gtru_arr/param.wavelength, tol=1/4)
            acc_pace_errs.append((pace_err, pace_jidx, pace_tp, pace_fp, pace_fn, conf_frame))
            acc_pala_errs.append((pala_err, pala_jidx, pala_tp, pala_fp, pala_fn, 0))

            print('PULM: RMSE: %0.4f, \tPrec.: %0.4f, \tRecall: %0.4f, \tJacc.: %0.4f' % (np.round(pace_err, 4), np.round(pace_p, 4), np.round(pace_r, 4), np.round(pace_jidx, 4)))
            print('PALA: RMSE: %0.4f, \tPrec.: %0.4f, \tRecall: %0.4f, \tJacc.: %0.4f' % (np.round(pala_err, 4), np.round(pala_p, 4), np.round(pala_r, 4), np.round(pala_jidx, 4)))
            
            if cfg.logging:
                wandb.log({
                    'PULM/RMSE': pace_err,
                    'PULM/Precision': pace_p,
                    'PULM/Recall': pace_r,
                    'PULM/Jaccard': pace_jidx,
                    'PULM/MeanFrameConfidence': conf_frame,
                    'PULM/TruePositive': pace_tp,
                    'PULM/FalsePositive': pace_fp,
                    'PULM/FalseNegative': pace_fn,
                    'frame': int(frame_idx+(dat_num-1)*cfg.frame_num),
                })
                wandb.log({
                    'PALA/RMSE': pala_err,
                    'PALA/Precision': pala_p,
                    'PALA/Recall': pala_r,
                    'PALA/Jaccard': pala_jidx,
                    'PALA/TruePositive': pala_tp,
                    'PALA/FalsePositive': pala_fp,
                    'PALA/FalseNegative': pala_fn,
                    'frame': int(frame_idx+(dat_num-1)*cfg.frame_num),
                })        

            if cfg.save_opt: np.savetxt((output_path / ('pace_frame_%s_%s.csv' % (str(dat_num).zfill(3), str(frame_idx).zfill(4)))), pace_arr, delimiter=',')
            if cfg.save_opt: np.savetxt((output_path / ('pala_frame_%s_%s.csv' % (str(dat_num).zfill(3), str(frame_idx).zfill(4)))), pala_arr, delimiter=',')
            if cfg.save_opt: np.savetxt((output_path / ('gtru_frame_%s_%s.csv' % (str(dat_num).zfill(3), str(frame_idx).zfill(4)))), gtru_arr, delimiter=',')

            if cfg.plt_cluster_opt:
                plt.rcParams.update({'font.size': 18})
                fig = plt.figure(figsize=(30/3*1.4, 15/3))
                gs = gridspec.GridSpec(1, 2)
                ax1 = plt.subplot(gs[0, 0])
                ax2 = plt.subplot(gs[0, 1])

                ax1.imshow(bmode, vmin=bmode_limits[0], vmax=bmode_limits[1], extent=extent, aspect=aspect**-1, origin='lower', cmap='gray')
                ax1.set_facecolor('#000000')
                #ax1.plot(rej_pts[:, 0], rej_pts[:, 1], '.', color='gray', label='rejected points', alpha=.2)
                #[ax1.text(rej_pts[i, 0], rej_pts[i, 1]+np.random.rand(1)*param.wavelength, s=str(rej_pts[i, 2]), color='orange') for i in range(len(rej_pts))]
                ax1.plot(xpos[~np.isnan(xpos)], zpos[~np.isnan(zpos)], marker='x', color='red', markersize=12, linestyle='', label='Ground-truth')
                ax1.plot((ref_xpos)*param.wavelength, (ref_zpos)*param.wavelength, marker='+', color='blue', markersize=12, linestyle='', label='Radial symmetry')
                ax1.plot(all_pts[:, 0], all_pts[:, 1], marker='1', color='cyan', markersize=12+2, linestyle='', label='Intersections', alpha=.6)
                ax1.plot(np.array(reduced_pts)[:, 0], np.array(reduced_pts)[:, 1], marker='.', color='orange', markersize=12, linestyle='', label='Centroid')
                ax1.plot([min(param.x), max(param.x)], [0, 0], color='gray', linewidth=8)   #, label='Transducer plane'
                ax1.set_ylim([0, max(param.z)])
                ax1.set_xlim([min(param.x), max(param.x)])
                ax1.legend(framealpha=1)
                #ax1.set_xlabel('Horizontal domain $x$ [m]')
                #ax1.set_ylabel('Vertical domain $z$ [m]')

                ax2.imshow(np.abs(iq_mat['IQ'][..., frame_idx]), cmap='gray')
                ax2.plot(xpos[~np.isnan(xpos)]/param.wavelength-PData['Origin'][0], zpos[~np.isnan(zpos)]/param.wavelength-PData['Origin'][2], 'rx', linestyle='', label='Ground-truth')
                ax2.plot(ref_xpos-PData['Origin'][0], ref_zpos-PData['Origin'][2], 'bx', label='Radial symmetry')
                ax2.plot(np.array(reduced_pts)[:, 0]/param.wavelength-PData['Origin'][0], np.array(reduced_pts)[:, 1]/param.wavelength-PData['Origin'][2], marker='1', color='cyan', linestyle='', label='Intersections')
                ax2.legend(framealpha=1)

                # switch all ticks off
                ax1.tick_params(top=False, bottom=False, left=False, right=False, labelleft=False, labelbottom=False)
                ax2.tick_params(top=False, bottom=False, left=False, right=False, labelleft=False, labelbottom=False)

                plt.tight_layout()

                # plot zoomed frame
                gt_pt_idx = 9-1
                gt_pt = np.array([xpos[~np.isnan(xpos)], zpos[~np.isnan(zpos)]])[:, gt_pt_idx]
                pa_pt_idx = np.argmin(abs(np.array([(ref_xpos)*param.wavelength, (ref_zpos)*param.wavelength]) - gt_pt[:, None]).sum(0))
                pa_pt = np.array([(ref_xpos)*param.wavelength, (ref_zpos)*param.wavelength])[:, pa_pt_idx]
                pt_idx = np.argmin(abs(np.array(reduced_pts).T - gt_pt[:, None]).sum(0))
                pt = np.array(reduced_pts)[pt_idx, :].T

                w = 2.5*param.wavelength
                axins1 = zoomed_inset_axes(ax1, zoom=6+4, loc='upper right')
                axins1.imshow(bmode, extent=extent, aspect=aspect**-1, origin='lower', cmap='gray')
                x1, x2, y1, y2 = gt_pt[0]-w, gt_pt[0]+w, gt_pt[1]-w*aspect, gt_pt[1]+w*aspect
                axins1.set_xlim(x1, x2)
                axins1.set_ylim(y1, y2)
                axins1.yaxis.get_major_locator().set_params(nbins=7)
                axins1.xaxis.get_major_locator().set_params(nbins=7)
                axins1.tick_params(top=False, bottom=False, left=False, right=False, labelleft=False, labelbottom=False)
                mark_inset(ax1, axins1, loc1=2, loc2=4, fc="none", ec='orange', lw=2)
                
                axins1.plot(gt_pt[0], gt_pt[1], 'rx', markersize=15, label='Ground truth')
                axins1.plot(pa_pt[0], pa_pt[1], 'b+', markersize=15, label='Radial symmetry')
                axins1.plot(all_pts[:, 0], all_pts[:, 1], marker='1', color='cyan', markersize=12, linestyle='', label='Intersections', alpha=.3)
                axins1.plot(pt[0], pt[1], '.', color='orange', markersize=15+2, label='Centroid')

                # style for frame
                sides_list = ['bottom', 'top', 'right', 'left']
                [axins1.spines[s].set_color('orange') for s in sides_list]
                [axins1.spines[s].set_linewidth(2) for s in sides_list]
        
                #for i, tx_gap_pts in enumerate(all_pts_list):
                    #ax1.plot(tx_gap_pts[:, 0], tx_gap_pts[:, 1], marker='.', linestyle='', color=['brown', 'pink', 'yellow', 'white', 'gray', 'cyan', 'green', 'blue'][i%8], label=str(cfg.tx_gaps[i]))
                    #axins1.plot(tx_gap_pts[:, 0], tx_gap_pts[:, 1], marker='.', linestyle='', color=['brown', 'pink', 'yellow', 'white', 'gray', 'cyan', 'green', 'blue'][i%8], label=str(cfg.tx_gaps[i]))
                fig.patch.set_alpha(0)  # transparency
                plt.savefig('./cluster_plot.pdf', format='pdf', backend='pdf', dpi=300, transparent=False)
                plt.show()

# total errors over all frames
pace_rmses = np.array(acc_pace_errs)[:, 0]
pala_rmses = np.array(acc_pala_errs)[:, 0]
pulm_rmse_mean = np.nanmean(pace_rmses)
pala_rmse_mean = np.nanmean(pala_rmses)
pulm_rmse_std = np.std(pace_rmses[~np.isnan(pace_rmses)])
pala_rmse_std = np.std(pala_rmses[~np.isnan(pala_rmses)])
pulm_jaccard_total = np.nansum(np.array(acc_pace_errs)[:, 2])/np.nansum(np.array(acc_pace_errs)[:, 2]+np.array(acc_pace_errs)[:, 3]+np.array(acc_pace_errs)[:, 4]) * 100
pala_jaccard_total = np.nansum(np.array(acc_pala_errs)[:, 2])/np.nansum(np.array(acc_pala_errs)[:, 2]+np.array(acc_pala_errs)[:, 3]+np.array(acc_pala_errs)[:, 4]) * 100
print('Total mean confidence: %s' % round(np.nanmean(np.array(acc_pace_errs)[:, -1]), 4))
print('Accumulated PULM RMSE: %s, Jacc.: %s' % (pulm_rmse_mean, pulm_jaccard_total))
print('Accumulated PALA RMSE: %s, Jacc.: %s' % (pala_rmse_mean, pala_jaccard_total))
if cfg.save_opt:
    np.savetxt(str(output_path / 'logged_errors.csv'), np.array(acc_pace_errs), delimiter=',')

    frames = load_ulm_data(data_path=str(output_path), expr='gtru')
    gtru_ulm_img, gtru_vel_map = render_ulm(frames, tracking=cfg.tracking, plot_opt=cfg.plt_frame_opt, cmap_opt=True, uint8_opt=False, gamma=cfg.gamma, srgb_opt=True)
    frames = load_ulm_data(data_path=str(output_path), expr='pala')
    pala_ulm_img, pala_vel_map = render_ulm(frames, tracking=cfg.tracking, plot_opt=cfg.plt_frame_opt, cmap_opt=True, uint8_opt=False, gamma=cfg.gamma, srgb_opt=True)
    frames = load_ulm_data(data_path=str(output_path), expr='pace')
    pace_ulm_img, pace_vel_map = render_ulm(frames, tracking=cfg.tracking, plot_opt=cfg.plt_frame_opt, cmap_opt=True, uint8_opt=False, gamma=cfg.gamma, srgb_opt=True)
    if cfg.logging:
        wandb.log({"gtru_ulm_img": wandb.Image(gtru_ulm_img)})
        wandb.log({"gtru_vel_img": wandb.Image(gtru_vel_map)})
        wandb.log({"pala_ulm_img": wandb.Image(pala_ulm_img)})
        wandb.log({"pala_vel_map": wandb.Image(pala_vel_map)})
        wandb.log({"pace_ulm_img": wandb.Image(pace_ulm_img)})
        wandb.log({"pace_vel_map": wandb.Image(pace_vel_map)})
if cfg.logging:
    wandb.summary['PULM/TotalRMSE'] = pulm_rmse_mean
    wandb.summary['PALA/TotalRMSE'] = pala_rmse_mean
    wandb.summary['PULM/TotalRMSEstd'] = pulm_rmse_std
    wandb.summary['PALA/TotalRMSEstd'] = pala_rmse_std
    wandb.summary['PULM/TotalJaccard'] = pulm_jaccard_total
    wandb.summary['PALA/TotalJaccard'] = pala_jaccard_total
    wandb.summary['PULM/TotalConfidence'] = np.nanmean(np.array(acc_pace_errs)[:, -1])
    wandb.save(str(output_path / 'logged_errors.csv'))
    if cfg.save_opt:
        wandb.summary['PALA/SSIM'] = structural_similarity(gtru_ulm_img, pala_ulm_img, channel_axis=2)
        wandb.summary['PULM/SSIM'] = structural_similarity(gtru_ulm_img, pace_ulm_img, channel_axis=2)
