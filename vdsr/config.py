import random

import numpy as np
import torch
from torch.backends import cudnn

# Random seed to maintain reproducible results
random.seed(0)
torch.manual_seed(0)
np.random.seed(0)
# Use GPU for training by default
device = torch.device("cuda", 0)
# Turning on when the image size does not change during training can speed up training
cudnn.benchmark = True
# Image magnification factor
upscale_factor = 2
# Current configuration parameter method
mode = "train"
# Experiment name, easy to save weights and log files
exp_name = "vdsr_baseline"

if mode == "train":
    # Dataset
    train_image_dir = "data/Urban100/VDSR/train"
    valid_image_dir = "data/Urban100/VDSR/valid"
    test_image_dir = "data/T91"

    image_size = 41
    batch_size = 16
    num_workers = 4

    # Incremental training and migration training
    start_epoch = 0
    resume = ""

    # Total num epochs
    epochs = 10

    # SGD optimizer parameter
    model_lr = 0.1
    model_momentum = 0.9
    model_weight_decay = 1e-4
    model_nesterov = False

    # StepLR scheduler parameter
    lr_scheduler_step_size = epochs // 4
    lr_scheduler_gamma = 0.1

    # gradient clipping constant
    clip_gradient = 0.01

    print_frequency = 200

if mode == "valid":
    # Test data address
    # sr_dir = f"results/test/{exp_name}"
    hr_dir = f"data/Set5/GTmod12"

    model_path = f"results/{exp_name}/best.pth.tar"
