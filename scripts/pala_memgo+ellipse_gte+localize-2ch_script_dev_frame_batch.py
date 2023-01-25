import sys
sys.path.append('../../')

import scipy.io
from scipy import signal
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Ellipse
import torch
import time
from sklearn.cluster import MeanShift, estimate_bandwidth
from pathlib import Path
from omegaconf import OmegaConf
import wandb

from multimodal_emg.batch_staged_memgo import batch_staged_memgo
from gte_intersect.ellipse import EllipseIntersection
from utils.pala_beamformer import pala_beamformer, decompose_frame
from utils.pala_error import rmse_unique
from utils.render_ulm import render_ulm
from utils.iq2rf import iq2rf

# tbd: replace t[echo_list] with batch_echo_array

def angle_amplitude_ratio(amplitude, xe_pos, pt):
    
    tx_pos = np.array([xe_pos, 0])

    # incidence angle w.r.t. perpendicular
    angle = np.arccos(np.array([0, 1]) @ np.array(pt-tx_pos) / (np.linalg.norm([0, 1]) * np.linalg.norm(pt-tx_pos)))#*180/np.pi

    # ratio with angle 
    fov_ratio = amplitude / (abs(angle)/np.pi*2+1)    # denom in [1, +inf] with orthogonal incidence being 1

    return fov_ratio, angle

def get_amp_grad(data, toa, phi_shift, ch_idx):

    sample_idx = time2sample(toa, phi_shift)
    sample_idx[(sample_idx >= len(data)) | (sample_idx < 0)] = 0
    sample_amp = data[sample_idx, ch_idx]
    sample_idx[(sample_idx >= len(data)-1) | (sample_idx+1 < 0)] = 0
    gradie_amp = data[sample_idx+1, ch_idx] - sample_amp

    return sample_amp, gradie_amp

def get_overall_phase_shift(data_arr, toas, phi_shifts, ch_idx, ch_gap, cch_sample, cch_grad):
    
    phi_shifts[-np.pi > phi_shifts] += +2*np.pi
    phi_shifts[+np.pi < phi_shifts] += -2*np.pi

    sample, grad = get_amp_grad(data_arr, toas, phi_shifts, ch_idx)
    phi_shifts[(sample < 0) & (cch_sample > 0) & (grad > 0) & (cch_grad < 0)] -= np.pi/2     
    phi_shifts[(sample > 0) & (cch_sample < 0) & (grad < 0) & (cch_grad > 0)] += np.pi/2

    sample, grad = get_amp_grad(data_arr, toas, phi_shifts, ch_idx)
    phi_shifts[(sample < 0) & (cch_sample > 0) & (((grad > 0) & (cch_grad > 0)) | ((grad < 0) & (cch_grad < 0)))] += np.pi/2
    phi_shifts[(sample > 0) & (cch_sample < 0) & (((grad > 0) & (cch_grad > 0)) | ((grad < 0) & (cch_grad < 0)))] -= np.pi/2

    sample, grad = get_amp_grad(data_arr, toas, phi_shifts, ch_idx)
    phi_shifts[(sample < 0) & (cch_sample < 0) & (grad > 0) & (cch_grad < 0)] += np.pi/2
    phi_shifts[(sample < 0) & (cch_sample < 0) & (grad < 0) & (cch_grad > 0)] -= np.pi/2

    sample, grad = get_amp_grad(data_arr, toas, phi_shifts, ch_idx)
    phi_shifts[(sample > 0) & (cch_sample > 0) & (grad > 0) & (cch_grad < 0)] -= np.pi/2
    phi_shifts[(sample > 0) & (cch_sample > 0) & (grad < 0) & (cch_grad > 0)] += np.pi/2

    # take full lambda phase shift into account
    phi_shifts[-np.pi > phi_shifts] += +2*np.pi
    phi_shifts[+np.pi < phi_shifts] += -2*np.pi

    return phi_shifts

script_path = Path(__file__).parent.resolve()

# load config
cfg = OmegaConf.load(str(script_path.parent / 'pulm_config.yaml'))

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

    #alt_fname = rel_path / 'Results' / 'PALA_InSilicoFlow_Tracks_multi_60dB.mat'#PALA_InSilicoFlow_Stats_multi60dB.mat'#PALA_InSilicoFlow_MatOut_multi_60dB.mat'
    #alt_mat = scipy.io.loadmat(alt_fname)

    res_fname = rel_path / 'Results' / 'matlab_wo_noise' / ('PALA_InSilicoFlow_raw_'+str(dat_num)+'.mat')   #.zfill(3)
    res_mat = scipy.io.loadmat(res_fname)

    rf_fname = rel_path / 'RF' / ('PALA_InSilicoFlow_RF'+str(dat_num).zfill(3)+'.mat')
    rf_mat = scipy.io.loadmat(rf_fname)

    if cfg.plt_comp_opt or cfg.plt_frame_opt:
        iq_fname = rel_path / 'IQ' / ('PALA_InSilicoFlow_IQ'+str(dat_num).zfill(3)+'.mat')
        iq_mat = scipy.io.loadmat(iq_fname)

    ulm_local_methods = [el[0] for el in res_mat['listAlgo'][0]]
    ulm_local_results = {m: arr for arr, m in zip(res_mat['Track_raw'][0], ulm_local_methods)}
    ulm_method = ulm_local_methods[-1]
    ref_pts = ulm_local_results[ulm_method]

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
    #vsource, width = InSilicoDataset.get_vsource(metadata, tx_idx=wave_idx)
    nonplanar_tdx = np.hypot((abs(vsource[0])-width/2)*(abs(vsource[0])>width/2), vsource[1])
    src_vec = np.array([vsource[0], vsource[1]])

    frame_start = 0
    for frame_batch_ptr in range(frame_start, frame_start+cfg.frame_num, frame_batch_size):

        # rf_iq_frame dimensions: frames x angles x samples x channels
        rf_iq_frames = np.array([decompose_frame(P, RFdata[..., frame_idx]) for frame_idx in range(frame_batch_ptr, frame_batch_ptr+frame_batch_size)])
        
        # convert IQ to RF data
        start = time.time()
        data_batch = iq2rf(np.hstack(rf_iq_frames[:, cfg.wave_idx, :, ::cfg.ch_gap]), mod_freq=param.f0, upsample_factor=cfg.enlarge_factor)
        print('Interpolation time: %s' % str(time.time()-start))

        # prepare variables for optimization
        data_batch = torch.tensor(data_batch, dtype=torch.double, device=cfg.device)
        t = torch.arange(0, len(data_batch[:, 0])/param.fs/cfg.enlarge_factor, 1/param.fs/cfg.enlarge_factor, device=data_batch.device, dtype=data_batch.dtype)

        # MEMGO optimization
        start = time.time()
        try:
            memgo_batch, result, conf_batch, echo_batch = batch_staged_memgo(data_batch, x=t, cfg=cfg, max_iter_per_stage=cfg.max_iter, print_opt=True)
            #memgo_batch, result, conf_batch, echo_batch = batched_memgo(data_batch, x=t, cfg=cfg, max_iter_per_stage=cfg.max_iter, print_opt=True)
        except torch._C._LinAlgError:
            continue
        print('MEMGO frame time: %s' % str((time.time()-start)/frame_batch_size))
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
            if cfg.plt_comp_opt or cfg.plt_frame_opt:
                iq_frame = iq_mat['IQ'][..., frame_idx]
                start = time.time()
                bmode = pala_beamformer(rf_iq_frames[frame_batch_idx, ...], param, mesh_x, mesh_z)
                print('Beamforming time: %s' % str(time.time()-start))
                bmode -= bmode.max()
                bmode_limits = [-60, 0] # [dB] scale

            # prevent echoes from being in negative time domain
            memgo_feats[memgo_feats[..., 1]<0, 1] = 0

            start = time.time()
            # select channel
            all_pts = []
            rej_pts = []
            acc_pts = []
            feats = []
            # iterate over transducer gaps (different baselines in parallax terms)
            for tx_gap in cfg.tx_gaps:

                # iterate over channels
                ch_idcs = np.hstack([np.arange(tx_gap+s, 16-tx_gap, tx_gap) for s in range(tx_gap)])

                # take only non-zero echo locations
                idcs_cch = (memgo_feats[ch_idcs, ..., 1] > 0)

                # take only most confident components
                #idcs_valid = np.argsort(conf[ch_idcs, idcs_cch])[-8:]   #idcs_valid = conf[ch_idx, idcs_cch] > np.nanmean(conf) /2   #
                comp_num = min(conf.shape[1], cfg.comp_max)
                idcs_valid = np.argsort(conf[ch_idcs, :], axis=-1)[:, -comp_num:]

                comps_cch = memgo_feats[ch_idcs[:, None], idcs_valid]

                # get central transducers ToA
                cch_dif = t[echo_list[ch_idcs[:, None], idcs_valid, 1].astype(int)] - comps_cch[..., 1] #if idx_cch < len(echo_list[ch_idx+000000]) else 0#t[np.argmax(emg_envelope_model(*comp_cch[:4], x=t.numpy()))] - comp_cch[1]
                toas_cch = (comps_cch[..., 1]+cch_dif+param.t0) * param.c + nonplanar_tdx
                cch_sample, cch_grad = get_amp_grad(data_arr, toas_cch, np.zeros(toas_cch.shape), ch_idcs[:, None])
                toas_cch -= 0.00001*cfg.shift_factor
                cch_cens = (np.array([param.xe[ch_idcs*cfg.ch_gap]*cfg.num_scale, np.zeros(ch_idcs.shape)]) + src_vec[:, None]*cfg.num_scale)/2

                spacer_vec = src_vec[:, None]*cfg.num_scale - np.array([param.xe[ch_idcs*cfg.ch_gap]*cfg.num_scale, np.zeros(ch_idcs.shape)])
                cch_vals = np.array([toas_cch*cfg.num_scale/2, ((toas_cch*cfg.num_scale)**2 - (spacer_vec**2).sum(0)[:, None])**.5 / 2])
                cch_vecs = spacer_vec / (spacer_vec**2).sum(0)**.5

                # get components from adjacent channels
                lch_cidx = np.argmin(abs(np.repeat(comps_cch[..., 1][..., None], memgo_feats.shape[1], axis=-1) - memgo_feats[ch_idcs-tx_gap, :, 1][:, None]), axis=-1)
                rch_cidx = np.argmin(abs(np.repeat(comps_cch[..., 1][..., None], memgo_feats.shape[1], axis=-1) - memgo_feats[ch_idcs+tx_gap, :, 1][:, None]), axis=-1)

                #matches = np.repeat(np.linspace(-1, 1, 3)[:, None], 2, axis=1).astype('int')
                #(np.repeat(lch_cidx[:, None], 3, axis=-1) + np.linspace(-1, 1, 3).astype('int')).flatten()
                lch_idcs = (np.repeat(lch_cidx[..., None], 3, axis=-1) + np.linspace(-1, 1, 3).astype('int')).reshape(lch_cidx.shape[0], -1)
                rch_idcs = (np.repeat(rch_cidx[..., None], 3, axis=-1) + np.linspace(-1, 1, 3).astype('int')).reshape(rch_cidx.shape[0], -1)
                echo_per_sch = 3
                
                # find comp index outliers (<0 or >num)
                # tbd = -1
                lch_idcs[(0>lch_idcs) | (lch_idcs>memgo_feats.shape[1]-1)] = 0#min(lch_idcs[(0<lch_idcs) | (lch_idcs<len(memgo_feats[ch_idx-tx_gap, ...]))])
                rch_idcs[(0>rch_idcs) | (rch_idcs>memgo_feats.shape[1]-1)] = 0#min(rch_idcs[(0<rch_idcs) | (rch_idcs<len(memgo_feats[ch_idx+tx_gap, ...]))])

                # prepare direct neighbour combinations
                comps_lch = memgo_feats[(ch_idcs-tx_gap)[:, None], lch_idcs]
                comps_rch = memgo_feats[(ch_idcs+tx_gap)[:, None], rch_idcs]

                lch_difs = t[echo_list[(ch_idcs-tx_gap)[:, None], lch_idcs, 1].astype(int)] - comps_lch[..., 1] #if idx_lch < len(echo_list[ch_idx-tx_gap]) else 0#t[np.argmax(emg_envelope_model(*comp_lch[:4], x=t.numpy()))] - comp_lch[1]
                rch_difs = t[echo_list[(ch_idcs+tx_gap)[:, None], rch_idcs, 1].astype(int)] - comps_rch[..., 1] #if idx_rch < len(echo_list[ch_idx+tx_gap]) else 0#t[np.argmax(emg_envelope_model(*comp_rch[:4], x=t.numpy()))] - comp_rch[1]

                # convert overall time-of-arrival distance
                toas_lch = (comps_lch[..., 1]+lch_difs+param.t0) * param.c + nonplanar_tdx
                toas_rch = (comps_rch[..., 1]+rch_difs+param.t0) * param.c + nonplanar_tdx

                # relative phase displacement
                phi_shift_cch = 0
                phi_shifts_lch = (comps_lch[..., 5] - np.repeat(comps_cch[..., 5], 3, axis=1))
                phi_shifts_rch = (comps_rch[..., 5] - np.repeat(comps_cch[..., 5], 3, axis=1))
                phi_shifts_lch = get_overall_phase_shift(data_arr, toas_lch, phi_shifts_lch, (ch_idcs-tx_gap)[:, None], cfg.ch_gap, np.repeat(cch_sample, 3, axis=1), np.repeat(cch_grad, 3, axis=1))
                phi_shifts_rch = get_overall_phase_shift(data_arr, toas_rch, phi_shifts_rch, (ch_idcs+tx_gap)[:, None], cfg.ch_gap, np.repeat(cch_sample, 3, axis=1), np.repeat(cch_grad, 3, axis=1))
                toas_lch -= phi_shifts_lch/(2*np.pi*param.fs) * param.c
                toas_rch -= phi_shifts_rch/(2*np.pi*param.fs) * param.c

                toas_lch -= 0.00001*cfg.shift_factor
                toas_rch -= 0.00001*cfg.shift_factor

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

                #print('1: %s' % str(time.time()-start))
                spts, _ = ell_intersector.get_intersection_multiple()
                #print('2: %s' % str(time.time()-start))
                spts /= cfg.num_scale
                pts_mask = (min(param.x) < spts[..., 0]) & (spts[..., 0] < max(param.x)) & (min(param.z) < spts[..., 1]) & (spts[..., 1] < max(param.z))
                pts = spts[pts_mask]
                pts_mask_num = np.any(pts_mask > 0, axis=1)

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

                    echo_pars = t[echo_list[par_ch_idcs, idx_pars, 1].astype(int)]  #np.array([t[echo_list[par_ch_idx, echo_idx, 1].astype(int)] if echo_idx < len(echo_list[par_ch_idx]) else 0 for (par_ch_idx, echo_idx) in zip(par_ch_idcs, idx_pars)])
                    par_difs = echo_pars - comp_pars[:, 1]
                    toa_pars = (comp_pars[:, 1]+par_difs+param.t0) * param.c + nonplanar_tdx
                    
                    # comp_cch indices to which each comp_par belongs to
                    #cch_idx_pars = np.concatenate([np.repeat(np.arange(len(ch_idcs)), echo_cch_num*echo_per_sch), np.repeat(np.arange(len(ch_idcs)), echo_cch_num*echo_per_sch)])[pts_mask_num]
                    cch_idx_pars = np.concatenate([np.repeat(np.arange(echo_cch_num), echo_per_sch), np.repeat(np.arange(echo_cch_num), echo_per_sch)])
                    #cch_idx_pars = np.repeat(cch_idx_pars[None, :], len(ch_idcs), axis=0) + np.repeat(np.arange(len(ch_idcs))[:, None], 6*8, axis=-1)
                    cch_idx_pars = np.repeat(cch_idx_pars[None, :], len(ch_idcs), axis=0)#.flatten() + 14*np.concatenate([np.repeat(np.arange(len(ch_idcs))[:, None], 3*8, axis=-1).flatten(), np.repeat(np.arange(len(ch_idcs))[:, None], 3*8, axis=-1).flatten()])
                    #cch_idx_pars = cch_idx_pars.flatten()[pts_mask_num]
                    s = np.array([comp_cch[cch_idx_par] for (comp_cch, cch_idx_par) in zip(comps_cch, cch_idx_pars)]).reshape(-1, 6)

                    phi_shift_pars = comp_pars[:, 5] - s[pts_mask_num, 5]#comps_cch.reshape(-1, 6)[cch_idx_pars, 5]####
                    cch_sample_par = np.repeat(np.concatenate([cch_sample, cch_sample], axis=-1).flatten(), echo_per_sch)[pts_mask_num]
                    cch_grad_par = np.repeat(np.concatenate([cch_grad, cch_grad], axis=-1).flatten(), echo_per_sch)[pts_mask_num]
                    phi_shift_pars = get_overall_phase_shift(data_arr, toa_pars, phi_shift_pars, par_ch_idcs, cfg.ch_gap, cch_sample_par, cch_grad_par)

                    toa_pars -= phi_shift_pars/(2*np.pi*param.fs) * param.c
                    dist_pars = abs((toa_pars-nonplanar_tdx)/param.c-param.t0 - mu_pars) * param.fs

                    valid = dist_pars < cfg.dist_par_threshold  # .5
                else:
                    dist_pars = np.ones(pts.shape[0])*float('NaN')
                    valid = np.ones(pts.shape[0], dtype=bool)

                if pts.size > 0: all_pts.append(np.array([pts[valid, 0], pts[valid, 1]]).T)
                if pts.size > 0: rej_pts.append(np.array([pts[~valid, 0], pts[~valid, 1]]).T)

                if cfg.plt_comp_opt:
                    
                    cch_idcs_flat = np.repeat(np.repeat(np.repeat(ch_idcs, comp_num).reshape(-1, comp_num), echo_per_sch, axis=1), 2, axis=0).flatten()[pts_mask_num]
                    sch_idcs_flat = np.repeat(np.dstack([ch_idcs-tx_gap, ch_idcs+tx_gap]).flatten(), echo_per_sch*comp_num)[pts_mask_num]
                    sch_comps = memgo_feats[sch_idcs_flat, idx_pars, :]
                    sch_phi_shifts = np.dstack([phi_shifts_lch, phi_shifts_rch]).swapaxes(1, -1).flatten()[pts_mask_num]

                    for k, pt in enumerate(pts):

                        if k == 351:
                            print('hold')

                        fig = plt.figure(figsize=(30, 15))
                        gs = gridspec.GridSpec(3, 2)
                        ax1 = plt.subplot(gs[:, 0])
                        ax2 = plt.subplot(gs[0, 1])
                        ax3 = plt.subplot(gs[1, 1])
                        ax4 = plt.subplot(gs[2, 1])

                        ax1.imshow(bmode[::-1, ...], vmin=bmode_limits[0], vmax=bmode_limits[1], extent=extent, aspect=aspect**-1, cmap='gray')
                        ax1.set_facecolor('#000000')
                        ax1.plot((ref_xpos)*param.wavelength, (ref_zpos)*param.wavelength, 'bx', label=ulm_method)
                        ax1.plot(xpos[~np.isnan(xpos)], zpos[~np.isnan(zpos)], 'rx', label='ground-truth')
                        ax1.plot([min(param.x), max(param.x)], [0, 0], color='gray', linewidth=5, label='Transducer plane')
                        ax1.set_ylim([0, max(param.z)])
                        ax1.set_xlim([min(param.x), max(param.x)])
                        ax1.set_xlabel('Lateral domain $x$ [m]')
                        ax1.set_ylabel('Axial distance $z$ [m]')

                        ax1.plot(pt[0], pt[1], '+', color='cyan', label='intersections')
                        ax1.text(pt[0], pt[1], s=str(dist_pars[k]), color='w')
                        ax1.legend()

                        # when is index k for left channel when of right? answer: pts_idcs
                        pts_idx = int(pts_idcs[k])
                        for j, (ax, cen, val, vec, color) in enumerate(zip([ax3]+[[ax2, ax4][pts_idx]], [cen_cens[:, pts_mask_num][:, k], adj_cens[:, pts_mask_num][:, k]], [cen_vals[:, pts_mask_num][:, k], adj_vals[:, pts_mask_num][:, k]], [cen_vecs[:, pts_mask_num][:, k], adj_vecs[:, pts_mask_num][:, k]], ['g']+[['b', 'y'][pts_idx]])):
                            
                            el_idx = cch_idcs_flat[k] if j==0 else sch_idcs_flat[k]
                            dmax = max([max(data_arr[:, cch_idcs_flat[k]]), max(data_arr[:, sch_idcs_flat[k]])])
                            dmin = min([min(data_arr[:, cch_idcs_flat[k]]), min(data_arr[:, sch_idcs_flat[k]])])
                                
                            # ellipse plot
                            major_axis_radius = np.linalg.norm(vec*val[0] / cfg.num_scale) 
                            minor_axis_radius = np.linalg.norm(vec*val[1] / cfg.num_scale) 
                            xz = np.array([cen[0], cen[1]]) / cfg.num_scale
                            # use float128 as small angles yield zero otherwise
                            vector_a = np.longdouble([(param.xe[el_idx*cfg.ch_gap] - vsource[0]), vsource[1]])
                            vector_b = np.longdouble([vsource[0], vsource[1]])
                            angle_deg = np.arccos(np.dot(vector_a, vector_b) / (np.linalg.norm(vector_a) * np.linalg.norm(vector_b))) / np.pi * 180
                            angle_deg *= -1*np.sign(param.xe[el_idx*cfg.ch_gap])
                            ell = Ellipse(xy=xz, width=2*minor_axis_radius, height=2*major_axis_radius, angle=angle_deg, edgecolor=color, fc='None')
                            ax1.add_artist(ell)

                            # plot detected mu echoes param
                            mus_samples = np.stack([(memgo_feats[el_idx, :, 1]) * param.fs * cfg.enlarge_factor,]*2)
                            mus_ys = np.stack([np.array([min(data_arr[:, el_idx]), max(data_arr[:, el_idx])]),]*memgo_feats.shape[-2]).T
                            ax.plot(mus_samples, mus_ys, color='gray')
                            ax.plot([[e[1] for e in echo_list[el_idx]], [e[1] for e in echo_list[el_idx]]], mus_ys[:, :len(echo_list[el_idx])], color='k')

                            # plot data and fitted result
                            ax.plot(np.abs(signal.hilbert(data_arr[:, el_idx])), label='Hilbert channel %s' % el_idx, color=color, alpha=.4)
                            ax.plot(data_arr[:, el_idx], label='Receive element %s' % el_idx, color=color)
                            ax.plot(result[el_idx, ...], label='Fitted %s' % el_idx, color='black', linestyle='dashed')

                            ax.set_xlim([0, len(data_arr[:, el_idx])])
                            #ax.set_xlim([0, max(param.z)])
                            ax.set_xlabel('Axial distance $z$ [samples]')
                            ax.set_ylabel('Amplitude $A(z)$ [a.u.]')
                            ax.grid(True)
                            ax.legend()

                            # plot rx trajectory
                            ax1.plot([param.xe[el_idx*cfg.ch_gap], pt[0]], [0, pt[1]], color, linestyle='dashed', label='receive trajectory of element %s' % el_idx)
                            ax1.legend()

                        # plot components
                        sax = [ax2, ax4][pts_idx]
                        sax.plot(np.stack([(sch_comps[k, 1] - sch_phi_shifts[k]/(2*np.pi*param.fs)) * param.fs * cfg.enlarge_factor,]*2), [dmin, dmax], color='red')
                        mu_cch = np.repeat(np.repeat(comps_cch[..., 1], echo_per_sch, axis=1), 2, axis=0).flatten()[pts_mask_num][k]
                        ax3.plot(np.stack([mu_cch * param.fs * cfg.enlarge_factor,]*2), [dmin, dmax], color='red')

                        #[ax2, ax4][~pt_idx].plot(np.stack([((toa_pars[k]-nonplanar_tdx)/param.c-param.t0) * param.fs * cfg.enlarge_factor,]*2), [min(result[par_ch_idcs[k], :]), max(result[par_ch_idcs[k], :])], color='pink', linestyle='dashdot', linewidth=2)
                        plt.tight_layout()
                        plt.show()

            all_pts = np.vstack(all_pts)
            rej_pts = np.vstack(rej_pts)

            #print('3: %s' % str(time.time()-start))
            ms.fit(all_pts[:, :2])
            labels = ms.labels_
            cluster_centers = ms.cluster_centers_

            #print('4: %s' % str(time.time()-start))

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

                if sum(l==labels) > cfg.cluster_number: reduced_pts.append(all_pts[l==labels][idx])

            #print('7: %s' % str(time.time()-start))

            print('Frame time: %s' % str(time.time()-start))

            gt_array = np.array([xpos, zpos]).T[(~np.isnan(xpos)) & (~np.isnan(zpos)), :]
            pace_array = np.array(reduced_pts)[:, :2] if np.array(reduced_pts).size > 0 else np.array([])
            pala_array = np.array([(ref_xpos)*param.wavelength, (ref_zpos)*param.wavelength]).T
            pace_err, pace_p, pace_r, pace_jidx, pace_tp, pace_fp, pace_fn = rmse_unique(pace_array/param.wavelength, gt_array/param.wavelength, tol=1/4)
            pala_err, pala_p, pala_r, pala_jidx, pala_tp, pala_fp, pala_fn = rmse_unique(pala_array/param.wavelength, gt_array/param.wavelength, tol=1/4)
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

            if cfg.save_opt: np.savetxt((output_path / ('pace_frame_%s_%s.csv' % (str(dat_num).zfill(3), str(frame_idx).zfill(4)))), pace_array, delimiter=',')
            if cfg.save_opt: np.savetxt((output_path / ('pala_frame_%s_%s.csv' % (str(dat_num).zfill(3), str(frame_idx).zfill(4)))), pala_array, delimiter=',')

            if cfg.plt_frame_opt:
                fig = plt.figure(figsize=(30, 15))
                gs = gridspec.GridSpec(1, 2)
                ax1 = plt.subplot(gs[0, 0])
                ax2 = plt.subplot(gs[0, 1])

                ax1.imshow(bmode[::-1, ...], vmin=bmode_limits[0], vmax=bmode_limits[1], extent=extent, aspect=aspect**-1, cmap='gray')
                ax1.set_facecolor('#000000')
                ax1.plot([min(param.x), max(param.x)], [0, 0], color='gray', linewidth=5, label='Transducer plane')
                ax1.plot(all_pts[:, 0], all_pts[:, 1], 'gx', label='all points', alpha=.2)
                ax1.plot(rej_pts[:, 0], rej_pts[:, 1], '.', color='gray', label='rejected points', alpha=.2)
                #[ax1.text(rej_pts[i, 0], rej_pts[i, 1]+np.random.rand(1)*param.wavelength, s=str(rej_pts[i, 2]), color='orange') for i in range(len(rej_pts))]
                ax1.plot((ref_xpos)*param.wavelength, (ref_zpos)*param.wavelength, 'bx', label=ulm_method)
                ax1.plot(xpos[~np.isnan(xpos)], zpos[~np.isnan(zpos)], 'rx', label='ground-truth')
                ax1.plot(np.array(reduced_pts)[:, 0], np.array(reduced_pts)[:, 1], 'c+', label='selected')
                ax1.set_ylim([0, max(param.z)])
                ax1.set_xlim([min(param.x), max(param.x)])
                ax1.set_xlabel('Lateral domain $x$ [m]')
                ax1.set_ylabel('Axial distance $z$ [m]')
                ax1.legend()

                ax2.imshow(np.abs(iq_mat['IQ'][..., frame_idx]), cmap='gray')
                ax2.plot(xpos[~np.isnan(xpos)]/param.wavelength-PData['Origin'][0], zpos[~np.isnan(zpos)]/param.wavelength-PData['Origin'][2], 'rx', label='ground-truth')
                ax2.plot(np.array(reduced_pts)[:, 0]/param.wavelength-PData['Origin'][0], np.array(reduced_pts)[:, 1]/param.wavelength-PData['Origin'][2], 'c+', label='selected')
                ax2.plot(ref_xpos-PData['Origin'][0], ref_zpos-PData['Origin'][2], 'bx', label=ulm_method)
                ax2.legend()
                for i, l in enumerate(labels_unique):
                    if sum(l==labels) > cfg.cluster_number: ax1.plot(all_pts[:, 0][l==labels], all_pts[:, 1][l==labels], marker='.', linestyle='', color=['brown', 'pink', 'yellow', 'white', 'gray', 'violet', 'green', 'blue'][i%8])

                plt.show()

# total errors over all frames
pace_rmses = np.array(acc_pace_errs)[:, 0]
pala_rmses = np.array(acc_pala_errs)[:, 0]
pulm_rmse_mean = np.nanmean(pace_rmses)
pala_rmse_mean = np.nanmean(pala_rmses)
pulm_rmse_std = np.std(pace_rmses[~np.isnan(pace_rmses)])
pala_rmse_std = np.std(pala_rmses[~np.isnan(pala_rmses)])
pulm_jaccard_total = np.array(acc_pace_errs)[:, 2].sum()/(np.array(acc_pace_errs)[:, 2]+np.array(acc_pace_errs)[:, 3]+np.array(acc_pace_errs)[:, 4]).sum() * 100
pala_jaccard_total = np.array(acc_pala_errs)[:, 2].sum()/(np.array(acc_pala_errs)[:, 2]+np.array(acc_pala_errs)[:, 3]+np.array(acc_pala_errs)[:, 4]).sum() * 100
print('Total mean confidence: %s' % round(np.nanmean(np.array(acc_pace_errs)[:, -1]), 4))
print('Accumulated PULM RMSE: %s, Jacc.: %s' % (pulm_rmse_mean, pulm_jaccard_total))
print('Accumulated PALA RMSE: %s, Jacc.: %s' % (pala_rmse_mean, pala_jaccard_total))
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
    np.savetxt(str(output_path / 'logged_errors.csv'), np.array(acc_pace_errs), delimiter=',')