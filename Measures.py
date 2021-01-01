# -*- coding: utf-8 -*-
"""
Created on Mon Jun  8 19:54:15 2020

@author: Steff
"""

from tqdm import tqdm
import torch
import numpy as np
import os
import imageio
import matplotlib.pyplot as plt
from Architecture import Normalize
from DetectionScore import Detector

################################################################
class GANMeasures:
    
    ############################################################
    def __init__(
            self,
            name,
            netD,
            netDs,
            netG,
            nz,
            fixed_noise_dim = 16,
            device = "cpu",
            spectral_loss = None,
            fid = None,
            fid_images = 10_000,
            fid_batch_size = 100
        ):
        
        self.name = name
        self.device = device
        self.netD = netD
        self.netDs = netDs
        self.netG = netG
        self.nz = nz
        self.spectral_loss = spectral_loss
        self.norm = Normalize()
        self.fid = fid
        self.fid_images = fid_images
        self.fid_batch_size = fid_batch_size
        self.fixed_noise = torch.randn(fixed_noise_dim, nz, 1, 1, device=self.device)
        self.fixed_noise_dim = fixed_noise_dim
        
        self.len_D = 0
        for p in netD.parameters(): self.len_D += 1
        self.len_G = 0
        for p in netG.parameters(): self.len_G += 1
        
        # measurements
        self.fake_images = []
        self.fid_scores = []
        self.average_profiles = []
        self.average_profile_losses_l1 = []
        
        self.others = {}
        
    ############################################################
    def measure(
            self,
            fake_images = True,
            fid = True,
            average_profile = True,
            spectral_loss = True
        ):
        
        with torch.no_grad():
            # set generator to evaluation only
            self.netG.eval()
            # set up progress bar
            with tqdm(  total = self.fid_images,
                        desc = "Measurements",
                        unit = "img",
                        leave = False  ) as pbar:
                # loop through batches
                for i in range(self.fid_images // self.fid_batch_size):
                    # generate batch of noise
                    noise = torch.randn(
                        self.fid_batch_size,self.nz,1,1,
                        device = self.device
                    )
                    # generate fake images
                    fake = self.netG(noise).detach()
                    fake = fake.clamp(min=0, max=1)
                    
                    if fid:
                        self.fid.fit_batch(self.norm(fake), batch_size=self.fid_batch_size)
                    if average_profile:
                        self.spectral_loss.fit_batch(fake)
                    
                    # update progress bar
                    pbar.update(len(fake))
                    
            # compute measurements
            if fid:
                self.fid_scores.append(
                    self.fid.compute_score_from_fit()
                )
                
            if average_profile:
                avg_profile = self.spectral_loss.complete_fit()
                self.average_profiles.append(
                    avg_profile
                )
                if spectral_loss:
                    self.average_profile_losses_l1.append(
                        self.spectral_loss.calc_from_profile(avg_profile.view(1,-1)).item()
                    )
                    
            # generate fake images        
            if fake_images:
                self._generate_fixed_fakes()
            
            # set generator back to training
            self.netG.train()
            
    ############################################################
    def best_models(self):
        models = set()
        
        if len(self.average_profile_losses_l1) > 0:
            models.add(
                self.average_profile_losses_l1.index(min(self.average_profile_losses_l1))
            )
            
        if len(self.fid_scores) > 0:
            models.add(
                self.fid_scores.index(min(self.fid_scores))
            )
        
        return models
    
    ############################################################
    def add_measure(self, name, val):
        if name not in self.others:
            self.others[name] = []
            
        self.others[name].append(val)
    
    ############################################################
    def _generate_fixed_fakes(self):
        fake_fixed = self.netG(self.fixed_noise).detach().cpu()
        self.fake_images.append(
            fake_fixed
        )
        
    ############################################################
    def fake_images_to_gif(
            self,
            save_path,
            name = None,
            plot_fid = True,
            plot_loss = True,
            plot_profile = True,
            rows = 4,
            cols = 4,
            alpha = 0.5
        ):
        
        def make_grid(imgs, rows=4, cols=4, padding=1):
            img_shape = list(imgs[0].shape)
            len_x = img_shape[1] + padding
            len_y = img_shape[2] + padding
            
            grid_shape = [img_shape[1], img_shape[2], img_shape[0]]
            grid_shape[0] = grid_shape[0]*cols + padding*(cols-1)
            grid_shape[1] = grid_shape[1]*rows + padding*(rows-1)
            
            grid = np.zeros((grid_shape))
            
            break_if = rows*cols
            
            for i in range(len(imgs)):
                if i >= break_if:
                    break
                
                img = np.transpose(imgs[i],(1,2,0))
                x = i % cols
                y = i // rows
                grid[x*len_x:(x+1)*len_x-padding, y*len_y:(y+1)*len_y-padding, :] = img
                
            return grid
        
        if name is None:
            name = self.name
        images = self.fake_images
        loss = self.average_profile_losses_l1
        loss = [float(l) for l in loss]
        fid = self.fid_scores
        profile = self.average_profiles
        real_profile = self.spectral_loss.mean.detach().cpu().numpy().flatten()
        
        file_temp = os.path.join(save_path,"temp.png")
        name_file = name.replace("/","").replace(":",",")
        file = os.path.join(save_path,f"{name_file}.mp4")
        
        size = [5,4]
        size[0] += int(plot_loss) + int(plot_profile) + int(plot_fid)
        epochs = len(images)
    
        with imageio.get_writer(file, mode="I", fps=8) as writer:
            for i in range(epochs):
                epoch = i+1
            
                plt.figure(figsize=(size[1]*1.95,size[0]*1.95))
                
                images_i = images[i].clamp(min=0, max=1)
                    
                images_i = images_i.cpu().numpy()

                img_grid = make_grid(images_i, rows, cols, 1)
                plt.subplot2grid(size, (1,0), rowspan=rows, colspan=cols)
                plt.axis("off")
                plt.imshow(img_grid)
            
                plt.subplot2grid(size,(0,0),rowspan=1,colspan=4)
                time = f"epoch {epoch:3.0f}"
                plt.text(
                    .5, .5, f"{name}\n{time}",
                    fontsize = 30,
                    ha = "center",
                    va = "center",
                    family = "monospace"
                )
                plt.axis("off")
                
                row = 5
                
                if plot_fid:
                    plt.subplot2grid(size, (row,0), rowspan=1, colspan=4)
                    row += 1
                    plt.plot(
                        range(1,epoch+1),
                        fid[:epoch],
                        marker="o"
                    )
                    plt.grid()
                    plt.xlim( (0, epochs+1) )
                    plt.ylim( (0, max(fid)*1.05) )
                    plt.legend(["FID"])
                
                if plot_loss:
                    plt.subplot2grid(size, (row,0), rowspan=1, colspan=4)
                    row += 1
                    plt.plot(
                        range(1,epoch+1),
                        loss[:epoch],
                        marker="o"
                    )
                    plt.grid()
                    plt.xlim( (0,epochs+1) )
                    plt.ylim( (0, max(loss)*1.05) )
                    plt.legend(["average spectral loss"])
            
                if plot_profile:
                    plt.subplot2grid(size, (row,0), rowspan=1, colspan=4)
                    plt.plot(real_profile, alpha=alpha)
                    profile_i = profile[i].detach().cpu().numpy().flatten()
                    plt.plot(profile_i, alpha=alpha)
                    plt.grid(axis="x")
                    plt.xticks(
                        range(len(real_profile)),
                        labels = [x if x%10==0 or x==len(real_profile)-1 else "" for x in range(len(real_profile))]
                    )
                    plt.legend(["real profile", "fake profile"])
                    
                plt.subplots_adjust(wspace=0, hspace=0)
                plt.tight_layout()
            
                plt.savefig(file_temp)
                plt.close()
            
                image = imageio.imread(file_temp)
                writer.append_data(image)
                
        return file
    
    ############################################################
    def get_measurements(self):
        m = {
            "GANMeasures.name":                         self.name,
            "GANMeasures.netG_str":                     str(self.netG),
            "GANMeasures.netD_str":                     str(self.netD),
            "GANMeasures.netDs_str":                    str(self.netDs),
            "GANMeasures.fake_images":                  self.fake_images,
            "GANMeasures.fid_scores":                   self.fid_scores,
            "GANMeasures.fid_real_stats":               self.fid.get_real_stats(),
            "GANMeasures.average_profiles":             self.average_profiles,
            "GANMeasures.real_profile":                 self.spectral_loss.mean,
            "GANMeasures.average_profile_losses_l1":    self.average_profile_losses_l1
        }
        for k, v in self.others.items():
            m[f"GANMeasures.{k}"] = v
        
        return m
    
    ############################################################
    def get_measurements_i(
            self,
            i,
            fake_images = True,
            fid_scores = True,
            average_profiles = True,
            average_profile_losses = True
        ):
        
        m_i = {}
        if len(self.fake_images) > i:
            m_i["GANMeasures.fake_images"] =                self.fake_images[i]
        if len(self.fid_scores) > i:
            m_i["GANMeasures.fid_scores"] =                 self.fid_scores[i]
        if len(self.average_profiles) > i:
            m_i["GANMeasures.average_profiles"] =           self.average_profiles[i]
        if len(self.average_profile_losses_l1) > i:
            m_i["GANMeasures.average_profile_losses_l1"] =  self.average_profile_losses_l1[i]
        
        m_other = {
            "GANMeasures.name":            self.name,
            "GANMeasures.netG_str":        str(self.netG),
            "GANMeasures.netD_str":        str(self.netD),
            "GANMeasures.netDs_str":       str(self.netDs),
            "GANMeasures.fid_real_stats":  self.fid.get_real_stats(),
            "GANMeasures.real_profile":    self.spectral_loss.mean,
            "GANMeasures.fixed_noise":     self.fixed_noise
        }
        for k, v in self.others.items():
            if len(v) == len(self.fid_scores):
                m_i[f"GANMeasures.{k}"] = v[i]
            else:
                m_other[f"GANMeasures.{k}"] = v
        
        return m_i, m_other
    
    ############################################################
    def compute_cloaking_score(
            self,
            dataloader,
            unnormalize,
            clamp
        ):
        
        reals = Detector._dataloader_to_profiles(
            dataloader = dataloader,
            img_size = self.spectral_loss.img_size,
            device = self.device
        )
        fakes = Detector._generator_to_profiles(
            G = self.netG,
            unnormalize = unnormalize,
            clamp = clamp,
            device = self.device,
            img_size = self.spectral_loss.img_size,
            nz = self.nz,
            N = len(reals),
            batch_size = 1_000
        )
        
        results = {}
        for epochs in [100, 1_000]:
            score, acc = Detector.from_profiles(
                reals = reals,
                fakes = fakes,
                device = self.device,
                batch_size = 1_000,
                epochs = epochs
            )
            results[epochs] = {
                "score": score,
                "acc": acc
            }
            
        return results
    
    ############################################################
    def measure_cloaking_scores(
            self,
            dataloader,
            unnormalize,
            clamp,
            checkpoint,
            epoch
        ):
        
        reals = Detector._dataloader_to_profiles(
            dataloader = dataloader,
            img_size = self.spectral_loss.img_size,
            device = self.device
        )
        
        for m in self.best_models():
            if m == epoch:
                G = self.netG
            else:
                G = checkpoint.load_g(m, self.device)
            
            fakes = Detector._generator_to_profiles(
                G = G,
                unnormalize = unnormalize,
                clamp = clamp,
                device = self.device,
                img_size = self.spectral_loss.img_size,
                nz = self.nz,
                N = len(reals),
                batch_size = 1_000
            )
            
            for epochs in [100, 1_000]:
                score, acc = Detector.from_profiles(
                    reals = reals,
                    fakes = fakes,
                    device = self.device,
                    batch_size = 1_000,
                    epochs = epochs
                )
                self.add_measure(
                    name = f"{m}:Cloak{epochs}",
                    val = score
                )
                
    ############################################################
    def print_cloaking_scores(self):
        for k in self.others.keys():
            if "Cloak" in k:
                print(f"{k}: {self.others[k]}")