# -*- coding: utf-8 -*-
"""
Created on Fri Jun  5 15:41:16 2020

@author: Steff
"""

from .inception import InceptionV3
import numpy as np
import os
import torch
from tqdm import tqdm
from scipy import linalg

################################################################
class FIDScore:
    
    f_cached_model = "inception.cache"
    f_cached_stats_real = "inception.stats_real.{}.cache"
    
    ############################################################
    def __init__(self, img_size, device="cpu"):
        self.device = device
        self.img_size = img_size
        
        self.model = InceptionV3(
            resize_input = True,
            normalize_input = False
        )
        
        # if os.path.isfile(FIDScore.f_cached_model):
        #     self.model = torch.load(
        #         FIDScore.f_cached_model,
        #         map_location = device
        #     )
        # else:
        #     self.model = InceptionV3(
        #         resize_input = True,
        #         normalize_input = False
        #     )
        #     torch.save(
        #         self.model,
        #         FIDScore.f_cached_model
        #     )
        
        self.model.eval()
        self.model.to(device)
        
        if os.path.isfile(FIDScore.f_cached_stats_real.format(img_size)): 
            self.mu_real, self.sigma_real = torch.load(
                FIDScore.f_cached_stats_real.format(img_size),
                map_location = device
            )
            self.is_fitted = True
        else:
            self.is_fitted = False
    
    ############################################################
    def fit_batch(self, data, batch_size=100):
        if not hasattr(self,"fitted_batches"):
            self.fitted_batches = []
            
        data_size = data.shape[0]
        dims = 2048
        
        pred_arr = np.empty((data_size, dims))
        
        i_arr = 0
        with tqdm(total=data_size, desc="Fit FID Batch", leave=False, unit="img") as pbar:
            for i in range(0, data_size, batch_size):
                start = i
                end = i + batch_size
                
                pred = self.model(
                    # self.up(
                        data[start:end]
                    # )
                )[0].cpu().numpy()
                pred_len = pred.shape[0]
                pred_arr[i_arr:i_arr+pred_len, :] = pred[:, :, 0, 0]
                i_arr += pred_len
                pbar.update(pred_len)
        
        self.fitted_batches.append(pred_arr)
        
    ############################################################
    def finalize_fit(self):
        fit = np.concatenate(self.fitted_batches)
        mu = np.mean(fit, axis=0)
        sigma = np.cov(fit, rowvar=False)
        del fit
        del self.fitted_batches
        
        return mu, sigma
    
    ############################################################
    def finalize_fit_real(self):
        self.mu_real, self.sigma_real = self.finalize_fit()
        torch.save(
            (self.mu_real, self.sigma_real),
            FIDScore.f_cached_stats_real.format(self.img_size)
        )
        self.is_fitted = True
        
    ############################################################
    def compute_score_from_data(self, data_fake, eps=1E-6):
        self.fit_batch(data_fake)
        return self.compute_score_from_fit(eps)
    
    ############################################################
    def compute_score_from_fit(self, eps=1E-6):
        mu2, sigma2 = self.finalize_fit()
        return self.compute_score(mu2, sigma2, eps)
        
    ############################################################
    def compute_score(self, mu2, sigma2, eps=1E-6):
        mu1, sigma1 = self.get_real_stats()

        diff = mu1 - mu2
    
        # Product might be almost singular
        covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
        if not np.isfinite(covmean).all():
            msg = ('fid calculation produces singular product; '
                   'adding %s to diagonal of cov estimates') % eps
            print(msg)
            offset = np.eye(sigma1.shape[0]) * eps
            covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))
    
        # Numerical error might give slight imaginary component
        if np.iscomplexobj(covmean):
            if not np.allclose(np.diagonal(covmean).imag, 0, atol=1E-3):
                m = np.max(np.abs(covmean.imag))
                raise ValueError('Imaginary component {}'.format(m))
            covmean = covmean.real
    
        tr_covmean = np.trace(covmean)
    
        return ( diff.dot(diff) +
                 np.trace(sigma1) +
                 np.trace(sigma2) -
                 2 * tr_covmean )
    
    ############################################################
    def get_real_stats(self):
        return self.mu_real, self.sigma_real