import sys
sys.path.append('../')

import numpy as np
import scipy
from pathlib import Path
from omegaconf import OmegaConf

from utils.render_ulm import render_ulm, load_ulm_data

script_path = Path(__file__).parent.resolve()

# load config
cfg = OmegaConf.load(str(script_path.parent / 'config_invivo.yaml'))

# override config with CLI
cfg = OmegaConf.merge(cfg, OmegaConf.from_cli())

run_name = 'kind-gorge-589'    #'twilight-universe-545' #'bright-grass-569'   #
output_path = Path(cfg.data_dir) / 'Results' / ('invivo_frames_'+run_name)

frames = load_ulm_data(data_path=str(output_path), expr='pace')

size = np.array([78,  118], dtype=int)  #None     #
origin = np.array([-59, 16, 0], dtype=int)# None   #
pala_origin = np.array([-72, 16, 0], dtype=int)# None   #

frame_end_idx = -1
frames = frames[:frame_end_idx]
pace_img_wo_tracks, _ = render_ulm(frames, tracking=None, plot_opt=False, cmap_opt=True, uint8_opt=False, gamma=cfg.gamma, srgb_opt=True, wavelength=9.856e-05, size=size, origin=origin)

# split Hungarian tracking into blocks to relieve memory
frames_per_block = 800
pace_img_wi_tracks = np.zeros((size[0]*10, size[1]*10, 3))
pala_img_wi_tracks = np.zeros((size[0]*10, size[1]*10, 3))
pala_img_wo_tracks = np.zeros((size[0]*10, size[1]*10, 3))
for i in range(len(frames)//frames_per_block+1):
    print('Block %s' % str(i))
    end_idx = (i+1)*frames_per_block if (i+1)*frames_per_block < len(frames) else len(frames)-1
    pace_block_wi_tracks, _ = render_ulm(frames[i*frames_per_block:end_idx], tracking='hungarian', plot_opt=False, cmap_opt=True, uint8_opt=False, gamma=cfg.gamma, srgb_opt=True, wavelength=9.856e-05, size=size, origin=origin, fps=1000)
    pace_img_wi_tracks += pace_block_wi_tracks

    # load PALA radial symmetry for comparison
    rs_fname = Path(cfg.data_dir) / 'Tracks' / ('PALA_InVivoRatBrain_Tracks'+str(i+1).zfill(3)+'.mat')
    rs_mat = scipy.io.loadmat(rs_fname)
    rs_pts = np.vstack(rs_mat['Track_raw'][0, 0][:, 0])
    del rs_mat
    rs_list = [rs_pts[rs_pts[:, -1]==i+1][:, :2][:, ::-1] for i in range(int(np.max(rs_pts[:, -1])))]
    pala_block_wo_tracks, _ = render_ulm(rs_list, tracking=None, plot_opt=False, cmap_opt=True, uint8_opt=False, gamma=cfg.gamma, srgb_opt=True, wavelength=1, size=size, origin=np.zeros(3), fps=1000)
    pala_img_wo_tracks += pala_block_wo_tracks
    pala_block_wi_tracks, _ = render_ulm(rs_list, tracking='hungarian', plot_opt=False, cmap_opt=True, uint8_opt=False, gamma=cfg.gamma, srgb_opt=True, wavelength=1, size=size, origin=np.zeros(3), fps=1000)
    pala_img_wi_tracks += pala_block_wi_tracks

    print('Avg. PACE points per frame %s' % str(np.mean([len(l) for l in frames[i*frames_per_block:end_idx]])))
    print('Avg. PALA points per frame %s' % str(np.mean([len(l) for l in rs_list])))

import wandb
wandb.init(project="pulm_renderer", name=run_name, config=cfg, group=None)
wandb.log({"img": wandb.Image(pace_img_wo_tracks/(pace_img_wo_tracks).max())})
wandb.log({"img_tracking": wandb.Image(pace_img_wi_tracks/(pace_img_wi_tracks).max())})
wandb.log({"img": wandb.Image(pala_img_wo_tracks/(pala_img_wo_tracks).max())})
wandb.log({"img_tracking": wandb.Image(pala_img_wi_tracks/(pala_img_wi_tracks).max())})
wandb.finish()
