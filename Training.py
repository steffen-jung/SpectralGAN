# -*- coding: utf-8 -*-
"""
Created on Sun Mar 22 00:04:15 2020

@author: Steff
"""

import os, random, argparse
from Checkpoint import Checkpoint

##############################################################################
#
#                              Parameter Settings
#
##############################################################################

binary = [0,1]
parser = argparse.ArgumentParser()
# Device to train on
parser.add_argument("--device",             type=str,   default="cuda:0")
# Increase the kernel size of the last upconvolutional layer in G
parser.add_argument("--large",              type=int,   choices=binary, default=1)
# Stack convolutional layer after the last upconvolution in G
parser.add_argument("--stacked",            type=int,   choices=binary, default=1)
# Name of the experiment (used as folder name)
parser.add_argument("--name",               type=str,   default="none")
# Folder to track experiments in
parser.add_argument("--experiments_folder", type=str,   default="/tmp/")
# Folder containing training images (in subfolders)
parser.add_argument("--data_folder",        type=str,   default="/data/")
# Number of epochs to train
parser.add_argument("--epochs",             type=int,   default=100)
# Size of the training data images
parser.add_argument("--img_size",           type=int,   default=64)
# Number of channels of the training images
parser.add_argument("--img_nc",             type=int,   default=3)
# Batch size during training
parser.add_argument("--batch_size",         type=int,   default=128)
# Set a random seed
parser.add_argument("--deterministic",      type=int,   default=0)
# Loss function for training
parser.add_argument("--loss",               type=str,   choices=["gan","wgan","lsgan"], default="gan")
# Gradient penalty for training
parser.add_argument("--gradient_penalty",   type=str,   choices=["none","wgan-gp"], default="none")
# Type of spectral discriminator
parser.add_argument("--d_spectral",         type=str,   choices=["none","linear","nonlinear"], default="none")
# How often to train the discriminators with each batch
parser.add_argument("--d_rounds",           type=int,   default=1)
# Scale the logits/error of the spectral discriminator
parser.add_argument("--g_spectral_ratio",   type=float, default=1.0)
# Method to combine discriminators
parser.add_argument("--g_combine",          type=str,   choices=["errors","logits"], default="errors")
# Checkpoint to continue training
parser.add_argument("--checkpoint",         type=str,   default="")
args = parser.parse_args()

# Number of training epochs
num_epochs = args.epochs

# Determine Discriminator/Generator training settings
combine_errors  = args.g_combine == "errors"
no_spectral     = args.d_spectral == "none"
if args.loss == "wgan" and args.gradient_penalty == "wgan-gp":
    args.d_rounds = 5

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
import numpy as np
from datetime import datetime
from tqdm import tqdm, trange
from DatasetPreload import Dataset
from SpectralLoss import SpectralLoss
from FID.FIDScore import FIDScore
from Measures import GANMeasures
from Architecture import Generator, Discriminator, SpectralDiscriminator, Normalize, Loss, Optimizer

if args.deterministic > 0:
    print("Training deterministically.")
    # Set random seed for reproducibility
    print(f"Random Seed: {args.deterministic}")
    random.seed(args.deterministic)
    torch.manual_seed(args.deterministic)
    np.random.seed(args.deterministic)
    torch.cuda.manual_seed(args.deterministic)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

print("Setting parameters.")

if len(args.checkpoint) == 0:
    # Folder for this run
    now = datetime.now()
    runfolder = now.strftime("%Y_%m_%d_%H_%M_%S")
    runfolder = f"{runfolder}_{args.name}"
    runfolder = os.path.join(args.experiments_folder, runfolder)
    os.mkdir(runfolder)
else:
    runfolder = args.checkpoint

print(f"Run folder: {runfolder}")

# Batch size during training
batch_size = args.batch_size
# Spatial size of training images. All images will be resized to this size using a transformer.
image_size = args.img_size
# Number of workers for dataloader
workers = 8
    
print(f"Dataloder workers: {workers}")

# Last upconvolutional layer
last_layer_large = args.large
last_layer_stacked = args.stacked

fid_track = True
fid_images = 10_000
fid_batch_size = fid_images // 100

print(f"Image Size: {image_size}")
print(f"Epochs: {num_epochs}")
print(f"Batch size: {batch_size}")

##############################################################################
#
#                              Dataset, FID
#
##############################################################################

device = args.device
print(f"Device: {device}")

print("Creating Dataset.")
# Create the dataset
dataset = Dataset(
    folder = args.data_folder,
    batch_size = batch_size,
    worker = workers
)
print(f"Dataset size: {dataset.length}")

# Custom weights initialization called on netG and netD
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm") != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

fid = FIDScore(image_size, device=device)
if not fid.is_fitted:
    print("Fitting FIDScore.")
    norm = Normalize()

    with tqdm(total=dataset.length, desc="Fitting FID", unit="img") as pbar:
        for batch, _ in dataset.dataloader:
            batch = norm(batch.to(device, non_blocking=True))
            fid.fit_batch(batch, batch_size=batch_size)
            pbar.update(len(batch))
        fid.finalize_fit_real()

##############################################################################
#
#                              Generator Code
#
##############################################################################
    
print("Create generator.")

# Size of z latent vector (i.e. size of generator input)
nz = 128
# Size of feature maps in generator
ngf = 32
# kernel size
kg = 4

# Create the generator
netG = Generator(
    image_size,
    nz = nz,
    ngf = ngf,
    kg = kg,
    last_layer_large = args.large,
    last_layer_stacked = args.stacked
).to(device)
print(netG)

number_parameters_G = netG.par_count()

# Initialize weights (mean=0, sd=0.2)
netG.apply(weights_init)

##############################################################################
#
#                              Discriminator
#
##############################################################################
    
print("Create discriminator.")

# Size of feature maps in discriminator
ndf = 64
# Kernel size discriminator
kd = 4

# Create the Discriminator
netD = Discriminator(
    image_size,
    ndf = ndf,
    kd = kd,
    nc = args.img_nc,
    batch_norm = not(args.loss == "wgan" and args.gradient_penalty == "wgan-gp")
).to(device)
print(netD)
number_parameters_D = netD.par_count()

netDs = None
number_parameters_Ds = 0
if not no_spectral:
    # Create the Discriminator
    netDs = SpectralDiscriminator(
        image_size,
        spectral = args.d_spectral,
    ).to(device)
    print(netDs)
    number_parameters_Ds = netDs.par_count()

# Initialize weights (mean=0, sd=0.2)
netD.apply(weights_init)

##############################################################################
#
#                              Losses, Measurements
#
##############################################################################

# Initialize loss
loss = Loss(
    loss = args.loss,
    gradient_penalty = args.gradient_penalty,
    device = device
)

# Spectral loss
loss_spectral = SpectralLoss(device=device, rows=image_size, cols=image_size, cache=True)
if not loss_spectral.is_fitted:
    with tqdm(total=dataset.length, desc="Fitting SpectralLoss", unit="img") as pbar:
        for batch, _ in dataset.dataloader:
            batch = batch.to(device, non_blocking=True)
            loss_spectral.fit_batch(batch)
            pbar.update(len(batch))
    loss_spectral.complete_fit_real(cache=True)

# Labels
real_label = 1
fake_label = 0

# Optimizers
optimizerD  = Optimizer(netD, args.loss, args.gradient_penalty).optimizer
optimizerDs = None
if not no_spectral:
    optimizerDs = Optimizer(netDs, args.loss, args.gradient_penalty).optimizer
optimizerG  = Optimizer(netG, args.loss, args.gradient_penalty).optimizer

# Measurements
measurements = GANMeasures(
    name = args.name,
    netD = netD,
    netDs = netDs,
    netG = netG,
    nz = nz,
    fixed_noise_dim = 16,
    device = device,
    spectral_loss = loss_spectral,
    fid = fid,
    fid_images = fid_images,
    fid_batch_size = fid_batch_size
)
measurements.add_measure("dataset_size", dataset.length)

##############################################################################
#
#                              Checkpoint
#
##############################################################################

chkpt_nets = {
    "netD": netD,
    "netG": netG
}
chkpt_optimizers = {
    "netD": optimizerD,
    "netG": optimizerG
}
if not no_spectral:
    chkpt_nets["netDs"] = netDs
    chkpt_optimizers["netDs"] = optimizerDs

checkpoint = Checkpoint(
    folder = runfolder,
    nets = chkpt_nets,
    optimizers = chkpt_optimizers,
    measurements = measurements
)

if len(args.checkpoint) > 0:
    print(f"Loading checkpoint: {args.checkpoint}")
    
    epoch_start = checkpoint.load(
        path = args.checkpoint,
        device = device
    ) + 1
    num_epochs += epoch_start
else:
    epoch_start = 0

##############################################################################
#
#                              Training Loop
#
##############################################################################

print(f"netD on device: {next(netD.parameters()).device}")
print(f"netD type: {type(netD)}")
print(f"netG on device: {next(netG.parameters()).device}")

##############################################################################

def train_D(real, fake, D, optimizer):
    # train D with real samples
    y_real = D(real)
    err_real = loss._loss(
        y = y_real,
        target = True
    )
    
    # train D with fake samples
    y_fake = D(fake.detach())
    err_fake = loss._loss(
        y = y_fake,
        target = False
    )
    
    gp = loss._gp(
        reals = real,
        fakes = fake,
        D = D
    )
    err = err_real + err_fake + gp
    err.backward()
    
    # optimize
    optimizer.step()
    
    # clip weights
    loss._weight_clipping(D)
    
    # return stats
    return ( y_real.detach().mean().item(),
             y_fake.detach().mean().item(),
             err.item(),
             err_real.item(),
             err_fake.item(),
             gp.item() )

##############################################################################

# train_G returns: y, y1, y2, err, err1, err2

def train_G_D1(fake, D1, optimizer, **kwargs):
    y = D1(fake)
    err = loss._loss(
        y = y,
        target = True
    )
    err.backward()
    
    # optimize
    optimizer.step()
    
    return ( 0.0,
             y.detach().mean().item(),
             0.0,
             0.0,
             err.item(),
             0.0 )

##############################################################################

def train_G_combine_logits(fake, a, b, D1, D2, optimizer, **kwargs):
    y1 = D1(fake).view(-1)
    y2 = D2(fake).view(-1)
    y = (a*y1 + b*y2) / (a+b)
    
    err = loss._loss(
        y = y,
        target = True
    )
    err.backward()
    
    # optimize
    optimizer.step()
    
    return ( y.detach().mean().item(),
             y1.detach().mean().item(),
             y2.detach().mean().item(),
             err.item(),
             0.0,
             0.0 )
    
##############################################################################

def train_G_combine_errors(fake, a, b, D1, D2, optimizer, **kwargs):
    y1 = D1(fake)
    y2 = D2(fake)
    
    err1 = loss._loss(
        y = y1,
        target = True
    )
    err2 = loss._loss(
        y = y2,
        target = True
    )
    err = (a*err1 + b*err2) / (a+b)
    err.backward()
    
    # optimize
    optimizer.step()

    return ( 0.0,
             y1.detach().mean().item(),
             y2.detach().mean().item(),
             err.item(),
             err1.item(),
             err2.item() )

##############################################################################

def training_loop():
    print("Starting Training Loop...", flush=True)
    iters = 0

    if no_spectral:
        train_G = train_G_D1
    else:
        if combine_errors:
            train_G = train_G_combine_errors
        else:
            train_G = train_G_combine_logits
    
    # For each epoch
    for epoch in trange(epoch_start, num_epochs, desc="Epoch", unit="epoch"):
        
        with tqdm(total=dataset.length, desc="Training iteration", unit="img", leave=False) as pbar:
            for batch, _ in dataset.dataloader:
                for i_d in range(args.d_rounds):
                    ############################
                    # (1) Update D networks
                    ###########################
                    netD.zero_grad()
                    if not no_spectral:
                        netDs.zero_grad()
    
                    # real images
                    real = batch.to(device, non_blocking=True)
                    b_size = real.size(0)
                    
                    # fake images
                    noise = torch.randn(b_size, nz, 1, 1, device=device)
                    fake = netG(noise)
                    
                    ### GAN discriminator ###
                    y1_real, y1_fake, err1, err1_real, err1_fake, gp1 = train_D(real, fake, netD, optimizerD)
                    
                    # measurements
                    measurements.add_measure("D1_y_real", y1_real)
                    measurements.add_measure("D1_y_fake", y1_fake)
                    measurements.add_measure("D1_err", err1)
                    measurements.add_measure("D1_err_real", err1_real)
                    measurements.add_measure("D1_err_fake", err1_fake)
                    measurements.add_measure("D1_gp", gp1)
                    
                    if not no_spectral:
                        ### Spectral discriminator ###
                        y2_real, y2_fake, err2, err2_real, err2_fake, gp2 = train_D(real, fake, netDs, optimizerDs)
                        
                        # measurements for saving
                        measurements.add_measure("D2_y_real", y2_real)
                        measurements.add_measure("D2_y_fake", y2_fake)
                        measurements.add_measure("D2_err", err2)
                        measurements.add_measure("D2_err_real", err2_real)
                        measurements.add_measure("D2_err_fake", err2_fake)
                        measurements.add_measure("D2_gp", gp2)
                    
                ############################
                # (2) Update G network
                ###########################
                netG.zero_grad()
                
                y, y1, y2, errG, errG1, errG2 = train_G(
                    fake = fake,
                    a = 1.0,
                    b = args.g_spectral_ratio,
                    D1 = netD,
                    D2 = netDs,
                    optimizer = optimizerG
                )
                
                # measurements
                measurements.add_measure("G_err", errG)
                measurements.add_measure("G_err_D1", errG1)
                measurements.add_measure("G_err_D2", errG2)
                measurements.add_measure("G_y", y)
                measurements.add_measure("G_y_D1", y1)
                measurements.add_measure("G_y_D2", y2)

                iters += 1
                pbar.update(b_size)
                
            # scores = {f"Cloak{epoch}_{s}" : val for epoch, results in scores.items() for s, val in results.items()}
            
            # for score, val in scores.items():
            #     measurements.add_measure(score, val)

        # Compute measurements
        
        measurements.measure()
        
        # Compute Cloaking Score
        
        if epoch == num_epochs-1:
            measurements.measure_cloaking_scores(
                dataloader = dataset.dataloader,
                unnormalize = False,
                clamp = True,
                checkpoint = checkpoint,
                epoch = epoch
            )
            
        # Checkpoint model
        
        checkpoint.save(epoch)
        
# torch.autograd.set_detect_anomaly(True)
training_loop()

measurements.print_cloaking_scores()