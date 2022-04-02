import yaml
import addict
import numpy as np
import os
import os.path as osp
from datetime import datetime
from glob import glob
import argparse

import torch
torch.autograd.set_detect_anomaly(True)

from models.cyclegan_model import CycleGANModel
from data.sample_dataloader import get_horse2zebra_train_dataloader, get_photo2monet_train_dataloader

# suppress all warnings
import warnings
warnings.filterwarnings("ignore")

# parse argument
parser = argparse.ArgumentParser(description='input the configure file path')
parser.add_argument('--opt', type=str, required=True, help='config file path')
args = parser.parse_args()
config_path = args.opt

# load configs
with open(config_path, 'r') as f:
    opt = yaml.load(f, Loader=yaml.FullLoader)
opt = addict.Dict(opt)

# modify config for training phase
# TODO: make it a function
opt['exp_name'] = os.path.basename(config_path[:-4])

# make dirs for save
save_dir = osp.join(opt['save_dir'], opt['exp_name'])
if os.path.exists(save_dir):
    timestamp = datetime.now().strftime("%Y%h%d%H%M")
    os.system(f'mv {save_dir} {save_dir}_archived_{timestamp}')
os.makedirs(save_dir, exist_ok=False)
os.system(f'cp {config_path} {save_dir}')

opt['path']['models'] = os.path.join(save_dir, 'ckpt')

# =========================================== #
#             DATA AND MODEL                  #
# =========================================== #

which_dataset = opt['datasets']['which_dataset']
dataroot = opt['datasets']['dataroot']
batch_size = opt['datasets']['batch_size']
img_size = opt['datasets']['img_size']

if which_dataset == 'horse2zebra':
    train_dataloader = get_horse2zebra_train_dataloader(dataroot, batch_size=batch_size, img_size=img_size)
elif which_dataset == 'photo2monet':
    train_dataloader = get_photo2monet_train_dataloader(dataroot, batch_size=batch_size, img_size=img_size)
else:
    raise NotImplementedError(f"Unrecognized dataset type : {which_dataset}")

model = CycleGANModel(opt)

# =========================================== #
#               TRAIN LOOP                    #
# =========================================== #

print(f"[TRAIN] start training ... ")

n_step = 0
for epoch in range(opt['train']['niter'] // len(train_dataloader)):
    for batch_data in train_dataloader:
        n_step += 1
        model.feed_data(batch_data)
        model.optimize_parameters(n_step)
        model.update_learning_rate(n_step, warmup_iter=opt["train"]["warmup_iter"])
        # print log
        if n_step % opt['logger']['print_freq'] == 0:
            loss_log = " ".join([f"{k}:{v:.4f}" for k, v in model.get_current_log().items()])
            print(f"{datetime.now()} epoch {epoch} n_step {n_step} losses: {loss_log}")

        # save visual eval result in training
        if n_step % opt['eval']['eval_interval'] == 0:
            model.save_current_visuals(os.path.join(save_dir, 'visuals'), n_step)

        # save models in training
        if n_step % opt['eval']['save_model_interval'] == 0:
            model.save(n_step)

print("[CycleGAN] train loop finished, everything ok")