import sys
sys.path.append('../')

import numpy as np
from pathlib import Path
from omegaconf import OmegaConf

from utils.render_ulm import render_ulm, load_ulm_data

script_path = Path(__file__).parent.resolve()

# load config
cfg = OmegaConf.load(str(script_path.parent / 'config_invivo.yaml'))

# override config with CLI
cfg = OmegaConf.merge(cfg, OmegaConf.from_cli())

run_name = 'kind-gorge-589'    #'bright-grass-569'   #
output_path = Path(cfg.data_dir) / 'Results' / ('invivo_frames_'+run_name)

frames = load_ulm_data(data_path=str(output_path), expr='pace')

frame_end_idx = -1

size = None     #np.array([78,  118], dtype=int)
origin = None   #np.array([-59,  0, 0], dtype=int)

ulm_img_wo_tracks, _ = render_ulm(frames[:frame_end_idx], tracking=None, plot_opt=False, cmap_opt=True, uint8_opt=False, gamma=cfg.gamma, srgb_opt=True, wavelength=9.856e-05, size=size, origin=origin)

ulm_img_wi_tracks, _ = render_ulm(frames[:frame_end_idx], tracking='hungarian', plot_opt=False, cmap_opt=True, uint8_opt=False, gamma=cfg.gamma, srgb_opt=True, wavelength=9.856e-05, size=size, origin=origin, fps=1000)

import wandb
wandb.init(project="pulm_renderer", name=run_name, config=cfg, group=None)
wandb.log({"img": wandb.Image(ulm_img_wo_tracks)})
wandb.log({"img": wandb.Image(ulm_img_wi_tracks)})
wandb.finish()
