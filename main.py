"""
Copyright (C) 2018 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""
from utils import get_loader, get_config
import argparse
import torch.backends.cudnn as cudnn
import torch
from solver import Solver
import os

parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, default='configs/face_folder.yaml', help='Path to the config file.')
parser.add_argument('--output_path', type=str, default='megan/samples')
parser.add_argument('--model_save_dir', type=str, default='megan/models')
parser.add_argument('--result_dir', type=str, default='megan/results')
parser.add_argument('--mode', type=str, default='train', choices=['train', 'test'])
parser.add_argument('--seed', type=int, default=10, help="random seed")
parser.add_argument('--num_style',type=int, default=10, help="number of styles to sample")
parser.add_argument('--log_path', type=str, default='megan/logs')

opts = parser.parse_args()

# 增加程序的运行速度
cudnn.benchmark = True

# Create directories if not exist.
if not os.path.exists(opts.log_path):
    os.makedirs(opts.log_path)
if not os.path.exists(opts.model_save_dir):
    os.makedirs(opts.model_save_dir)
if not os.path.exists(opts.output_path):
    os.makedirs(opts.output_path)
if not os.path.exists(opts.result_dir):
    os.makedirs(opts.result_dir)

# Load experiment setting
config = get_config(opts.config)
#为当前GPU设置随机种子，使得结果是确定的
torch.manual_seed(opts.seed)
#为所有的GPU设置种子
torch.cuda.manual_seed(opts.seed)


face_loader = None
face_loader = get_loader(config['data_root'],
                                 config['face_crop_size'], config['face_image_size'], config['batch_size'],
                                 'train', config['num_workers'])

solver = Solver(face_loader, config, opts)
if opts.mode == 'train':
    solver.train()
elif opts.mode == 'test':
    solver.test()