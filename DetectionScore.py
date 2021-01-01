# -*- coding: utf-8 -*-
"""
Created on Sun Jul  5 14:49:58 2020

@author: Steff
"""

import tqdm
import torch
import torchvision.datasets as dset
import torchvision.transforms as transforms
import numpy as np

class Detector(torch.nn.Module):
    def __init__(self, vector_size):
        super(Detector, self).__init__()
        self.lr = torch.nn.Linear(vector_size, 1)
        
    ##########################################################################
    @staticmethod
    def from_dataloader_and_generator(
            dataloader,
            G,
            unnormalize,
            clamp,
            device = "cpu",
            img_size = 64,
            nz = 128,
            batch_size = 1_000,
            epochs = 10_000
        ):
        
        profiles_real = Detector._dataloader_to_profiles(
            dataloader = dataloader,
            img_size = img_size,
            device = device
        )
        profiles_fake = Detector._generator_to_profiles(
            G = G,
            unnormalize = unnormalize,
            clamp = clamp,
            device = device,
            img_size = img_size,
            nz = nz,
            N = len(profiles_real),
            batch_size = batch_size
        )
        
        return Detector.from_profiles(
            reals = profiles_real,
            fakes = profiles_fake,
            device = device,
            batch_size = batch_size,
            epochs = epochs
        )
        
    ##########################################################################
    @staticmethod
    def from_folder_and_generator(
            folder_reals,
            G,
            unnormalize,
            clamp,
            device = "cpu",
            img_size = 64,
            nz = 128,
            batch_size = 1_000,
            workers = 4,
            epochs = 10_000
        ):
        
        profiles_real = Detector._folder_to_profiles(
            folder = folder_reals,
            device = device,
            batch_size = batch_size,
            workers = workers
        )
        profiles_fake = Detector._generator_to_profiles(
            G = G,
            unnormalize = unnormalize,
            clamp = clamp,
            device = device,
            img_size = img_size,
            nz = nz,
            N = len(profiles_real),
            batch_size = batch_size
        )
        
        return Detector.from_profiles(
            reals = profiles_real,
            fakes = profiles_fake,
            device = device,
            batch_size = batch_size,
            epochs = epochs
        )
    
    ##########################################################################
    @staticmethod
    def from_folders(
            folder_a,
            folder_b,
            device = "cpu",
            batch_size = 10_000,
            workers = 4,
            epochs = 10_000
        ):
        
        profiles_a = Detector._folder_to_profiles(
            folder = folder_a,
            device = device,
            batch_size = batch_size,
            workers = workers
        )
        profiles_b = Detector._folder_to_profiles(
            folder = folder_b,
            device = device,
            batch_size = batch_size,
            workers = workers
        )
        
        return Detector.from_profiles(
            reals = profiles_a,
            fakes = profiles_b,
            device = device,
            batch_size = batch_size,
            epochs = epochs
        )
    
    ##########################################################################
    @staticmethod
    def _folder_to_profiles(
            folder,
            device = "cpu",
            batch_size = 1_000,
            workers = 4
        ):
        
        dataset = dset.ImageFolder(
            root = folder,
            transform = transforms.Compose([
                transforms.ToTensor()
            ])
        )
        
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size = batch_size,
            shuffle = False,
            pin_memory = False,
            num_workers = workers
        )
        
        img_size = dataset[0][0].shape[1]
        
        return Detector._dataloader_to_profiles(
            dataloader = dataloader,
            img_size = img_size,
            device = device
        )
    
    ##########################################################################
    @staticmethod
    def _dataloader_to_profiles(
            dataloader,
            img_size,
            device = "cpu"
        ):
        
        from SpectralLoss import SpectralLoss

        spectral = SpectralLoss(
            device = device,
            rows = img_size,
            cols = img_size
        )
        
        profiles = []
        with torch.no_grad():
            with tqdm.tqdm(total=len(dataloader), desc="Folder to profiles", unit="img") as pbar:   
                for batch, _ in dataloader:
                    batch = batch.to(device)
                    profiles.append(
                        spectral.spectral_vector(batch)
                    )
                    pbar.update(len(batch))
                    
            profiles = torch.cat(profiles)
            
        del spectral
        
        return profiles
    
    ##########################################################################
    @staticmethod
    def _generator_to_profiles(
            G,
            unnormalize,
            clamp,
            device = "cpu",
            img_size = 64,
            nz = 128,
            N = 70_000,
            batch_size = 1_000
        ):
        
        from SpectralLoss import SpectralLoss
        from Architecture import Unnormalize
        
        un = Unnormalize()
        spectral = SpectralLoss(
            device = device,
            rows = img_size,
            cols = img_size
        )
        
        profiles = []        
        noise = torch.randn(N, nz, 1, 1, device=device)
        batches = int( np.ceil(N / batch_size) )
        G.eval()
        with torch.no_grad():
            with tqdm.tqdm(total=N, desc="Generator to profiles", unit="img") as pbar:   
                for i in range(batches):
                    batch = G(noise[i*batch_size : (i+1)*batch_size])
                    if unnormalize:
                        batch = un(batch)
                    if clamp:
                        batch.clamp_(min=0, max=1)
                    profiles.append(
                        spectral.spectral_vector(batch)
                    )
                    pbar.update(len(batch))
            profiles = torch.cat(profiles)
        
        return profiles
    
    ##########################################################################
    @staticmethod
    def from_profiles(
            reals,
            fakes,
            device = "cpu",
            batch_size = 1_000,
            epochs  = 10_000
        ):
        
        assert reals.shape[1] == fakes.shape[1]
        detector = Detector(reals.shape[1]).to(device)
      
        data_profiles = torch.cat((reals, fakes)).to(device)
        data_len = len(data_profiles)
        
        data_target = torch.cat((
            torch.ones(len(reals)),
            torch.zeros(len(fakes))
        )).to(device)
        
        opt = torch.optim.Adam(detector.parameters())
        loss = torch.nn.BCEWithLogitsLoss(reduction="sum")
        
        # Train for {epochs} epochs
        for epoch in tqdm.trange(1, epochs+1, unit="epoch"):
            shuffle = np.random.permutation(range(data_len))
            i = 0
            
            while i < data_len:
                opt.zero_grad()
                
                j = i + batch_size
                idx = shuffle[i : j]
                batch = data_profiles[idx]
                target = data_target[idx]
                i = j
    
                y = detector(batch).squeeze()
                l = loss(y,target)
                l.backward()
                opt.step()
                
        # Compute score
        wrong = 0
        with torch.no_grad():
            i = 0
            while i < data_len:
                j = i + batch_size
                batch = data_profiles[i : j]
                target = data_target[i : j]
                i = j
                
                y = detector(batch).squeeze()
                wrong += np.sum( np.abs( (y.detach().cpu().numpy()>0).astype(np.float) - target.cpu().numpy() ) )
                
        acc   = 1.0 -  wrong / data_len
        score = 1.0 - abs( (acc - .5) / .5 )
        
        return score, acc
        
    ##########################################################################
    def forward(self, x):
        return self.lr(x)