# Importing required libraries
import sys
import json
import time
# from nni.experiment import Experiment
# from model import ResNet, ResBlock, train, test, device, fine_tune
# from torch.optim import Adam
import torch
from nni.compression.pytorch.pruning import L1NormPruner, FPGMPruner
from nni.compression.pytorch.speedup import ModelSpeedup
from model import VDSR
import train as train_def
from torch.cuda import amp
import os
from torch.utils.tensorboard import SummaryWriter
import config


epochs = 10

if __name__ == '__main__':
    device = 'cuda'

    model_chkpt = torch.load('vdsr-TB291-fef487db.pth.tar',map_location='cuda:0')
    model = VDSR().to(device)
    model.load_state_dict(model_chkpt['state_dict'])
    
    optimizer = train_def.define_optimizer(model)
    optimizer.load_state_dict(model_chkpt['optimizer'])

    train_prefetcher, valid_prefetcher, test_prefetcher = train_def.load_dataset()
    psnr_criterion, pixel_criterion = train_def.define_loss()
    scaler = amp.GradScaler()
    writer = SummaryWriter(os.path.join("samples", "logs", "vdsr_pruning"))

    print("ORIGINAL UN-PRUNED MODEL: \n\n", model, "\n\n")
    start_time = time.time()

    for epoch in range(epochs):
        train_def.train(model, train_prefetcher, psnr_criterion, pixel_criterion, optimizer, epoch, scaler, writer)
        _ = train_def.validate(model, valid_prefetcher, psnr_criterion, epoch, writer, "Valid")
    psnr = train_def.validate(model, test_prefetcher, psnr_criterion, epoch, writer, "Test")
    # psnr = train_def.validate(model, test_prefetcher, psnr_criterion, 1, writer, "Test")
    print(f"******\nOriginal PSNR: {psnr}\n*****")
    print(f"Original model training time: {time.time() - start_time}")
    # torch.save(model, f'./unpruned_model.torch')
    torch.save({"epoch": epoch + 1,
                "best_psnr": psnr,
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict()},
                f"vdsr_unpruned_{config.upscale_factor}.pth.tar")
    print('Unpruned model saved')

    configuration_list = [{
        'sparsity_per_layer': 0.4,
        'op_types': ['Conv2d']
    }, {
        'exclude': True,
        'op_names': ['conv1', 'conv2']
    }]
    # Defining the pruner to be used
    pruner = L1NormPruner(model, configuration_list)
    
    print(f"PRUNER WRAPPED MODEL WITH L1 NormPruner: \n\n{model}\n\n")

    # Next, compressing the model and generating masks
    _, masks = pruner.compress()
    # show the masks sparsity
    for name, mask in masks.items():
        print(name, ' sparsity : ', '{:.2}'.format(mask['weight'].sum() / mask['weight'].numel()), "\n")

    # Need to unwrap the model before speeding-up.
    pruner._unwrap_model()

    dummy_input = torch.rand(64, 1, 3,3)
    ModelSpeedup(model, dummy_input.to(device), masks).speedup_model()

    print(f"PRUNED MODEL WITH L1 NormPruner: \n\n{model}\n\n")

    
    # Running the pre-training stage with pruned model
    
    start_time = time.time()
    for epoch in range(epochs):
        train_def.train(model, train_prefetcher, psnr_criterion, pixel_criterion, optimizer, epoch, scaler, writer)
        _ = train_def.validate(model, valid_prefetcher, psnr_criterion, epoch, writer, "Valid")
    psnr = train_def.validate(model, test_prefetcher, psnr_criterion, epoch, writer, "Test")
    print(f"\n********\nPruned PSNR: {psnr}\n********\n")
    print(f"Pruned model training time: {time.time() - start_time}")
    torch.save({"epoch": epoch + 1,
                "best_psnr": psnr,
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict()},
                f"vdsr_pruned_{config.upscale_factor}.pth.tar")
    print('Pruned model saved')
    
    model.eval()
    torch.onnx.export(
        model,
        dummy_input.to(device),
        f"vdsr_pruned_{config.upscale_factor}.onnx",
        export_params=True,
        do_constant_folding=True,
        input_names=["data"],
        output_names=["output"],
        opset_version=11
    )