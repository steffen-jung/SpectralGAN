# -*- coding: utf-8 -*-
"""
Created on Thu Apr  2 16:10:37 2020

@author: Steff
"""

import torch
import numpy as np
import os

################################################################
class SpectralLoss(torch.nn.Module):
    
    f_cache = "spectralloss.{}.cache"
    
    ############################################################
    def __init__(self, rows=64, cols=64, eps=1E-8, device=None, cache=False):
        super(SpectralLoss, self).__init__()
        self.img_size = rows
        self.eps = eps
        ### precompute indices ###
        # anticipated shift based on image size
        shift_rows = int(rows / 2)
        # number of cols after onesided fft
        cols_onesided = int(cols / 2) + 1
        # compute radii: shift columns by shift_y
        r = np.indices((rows,cols_onesided)) - np.array([[[shift_rows]],[[0]]])
        r = np.sqrt(r[0,:,:]**2+r[1,:,:]**2)
        r = r.astype(int)
        # shift center back to (0,0)
        r = np.fft.ifftshift(r,axes=0)
        ### generate mask tensors ###
        # size of profile vector
        r_max = np.max(r)
        # repeat slice for each radius
        r = torch.from_numpy(r).expand(
            r_max+1,-1,-1
        )
        radius_to_slice = torch.arange(r_max+1).view(-1,1,1)
        # generate mask for each radius
        mask = torch.where(
            r==radius_to_slice,
            torch.tensor(1,dtype=torch.float),
            torch.tensor(0,dtype=torch.float)
        )
        # how man entries for each radius?
        mask_n = torch.sum(mask,axis=(1,2))
        mask = mask.unsqueeze(0) # add batch dimension
        # normalization vector incl. batch dimension
        mask_n = (1/mask_n.to(torch.float)).unsqueeze(0)
        self.criterion_l1 = torch.nn.L1Loss(reduction="sum")
        self.r_max = r_max
        self.vector_length = r_max+1
        
        self.register_buffer("mask", mask)
        self.register_buffer("mask_n", mask_n)
        
        if cache and os.path.isfile(SpectralLoss.f_cache.format(self.img_size)):
            self._load_cache()
        else:
            self.is_fitted = False
            self.register_buffer("mean", None)
            
        if device is not None:
            self.to(device)
        self.device = device
            
    ############################################################
    def _save_cache(self):
        torch.save(self.mean, SpectralLoss.f_cache.format(self.img_size))
        self.is_fitted = True
        
    ############################################################
    def _load_cache(self):
        mean = torch.load(
            SpectralLoss.f_cache.format(self.img_size),
            map_location = self.mask.device
        )
        self.register_buffer("mean", mean)
        self.is_fitted = True
        
    ############################################################
    #                                                          #
    #               Spectral Profile Computation               #
    #                                                          #
    ############################################################
    
    ############################################################
    def fft(self,data):
        if len(data.shape) == 4 and data.shape[1] == 3:
            # convert to grayscale
            data =  0.299 * data[:,0,:,:] + \
                    0.587 * data[:,1,:,:] + \
                    0.114 * data[:,2,:,:]
        fft = torch.rfft(data,2,onesided=True)
        # abs of complex
        fft_abs = torch.sum(fft**2,dim=3)
        fft_abs = fft_abs + self.eps
        fft_abs = 20*torch.log(fft_abs)
        
        return fft_abs
    
    ############################################################
    def spectral_vector(self, data):
        """Assumes first dimension to be batch size."""
        fft = self.fft(data) \
                .unsqueeze(1) \
                .expand(-1,self.vector_length,-1,-1) # repeat img for each radius

        # apply mask and compute profile vector
        profile = (fft * self.mask).sum((2,3))
        # normalize profile into [0,1]
        profile = profile * self.mask_n
        profile = profile - profile.min(1)[0].view(-1,1)
        profile = profile / profile.max(1)[0].view(-1,1)
        
        return profile
    
    ############################################################
    def avg_profile(self, data):
        profile = self.spectral_vector(data)
        return profile.mean(0)
    
    ############################################################
    def avg_profile_batched(self, data, batch_size=1024, dtype=torch.double):
        i = 0
        v_total = torch.zeros(
            1,self.vector_length,
            dtype=dtype,
            device=self.device
        )
        while i < len(data):
            i_next = i + batch_size
            v = torch.sum(
                self.spectral_vector(data[i:i_next]),
                dim=0
            )
            v_total += v
            i = i_next
        return v_total / len(data)
    
    ############################################################
    def avg_profile_and_sd(self, data, batch_size=1024):
        if len(data) < batch_size:
            profile = self.spectral_vector(data)
            return profile.mean(0), profile.std(0)
        else:
            i = 0
            v_total = []
            while i < len(data):
                i_next = i + batch_size
                v_total.append(
                    self.spectral_vector(data[i:i_next])
                )
                i = i_next
            
            v = torch.cat(v_total)
            return v.mean(0), v.std(0)
    
    ############################################################
    def fit_batch(self, batch):
        if not hasattr(self,"batches"):
            self.batches = []
            self.batches_size = 0
        
        v = np.sum(
            self.spectral_vector(batch).detach().cpu().numpy(),
            axis = 0
        ).reshape((1,-1))
        
        self.batches_size += len(batch)
        self.batches.append(v)
        
    ############################################################
    def complete_fit(self):
        total = np.sum(
            np.concatenate(self.batches),
            axis = 0
        )
        mean = torch.from_numpy(total / self.batches_size)
        if self.device is not None:
            mean = mean.to(self.device)
        del self.batches
        del self.batches_size
        
        return mean
        
    ############################################################
    def complete_fit_real(self, cache=False):
        self.mean = self.complete_fit()

        if cache:
            self._save_cache()
    
    ############################################################
    def fit(self, data, batch_size=1024, cache=False):
        self.mean = self.avg_profile_batched(data,batch_size)
        self.register_buffer("mean",self.mean)
        
        if cache:
            self._save_cache()
            
    ############################################################
    #                                                          #
    #               Spectral Profile Computation               #
    #                                                          #
    ############################################################
    
    ############################################################
    def calc_from_profile(self, profile):
        batch_size = profile.shape[0]
        target = self.mean.expand(batch_size,-1)
        
        return self.criterion_l1(profile, target)