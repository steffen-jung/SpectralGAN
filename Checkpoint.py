# -*- coding: utf-8 -*-
"""
Created on Tue Jul 14 16:17:27 2020

@author: Steff
"""

import os
import torch
from Measures import GANMeasures
from Architecture import Generator

class Checkpoint:
    
    ###########################################################################
    def __init__(
            self,
            folder       : str,
            nets         : dict,
            optimizers   : dict,
            measurements : GANMeasures
        ):
        
        self.folder = folder
        self.nets = nets
        self.optimizers = optimizers
        self.measurements = measurements
        self.keep_models = set()
        # assertions
        for key in nets:
            if key not in optimizers:
                raise RuntimeError(f"'{key}' not in optimizers dict.")
            if not hasattr(nets[key],"to_checkpoint"):
                raise RuntimeError(f"net '{key}' is not implementing to_checkpoint method.")
    
    ###########################################################################
    def save(self, epoch):
        path_model = os.path.join(
            self.folder, "{}.model"
        )
        path_measures = os.path.join(
            self.folder, "{}.measures"
        )
        
        # Checkpoint model
        
        save = {
            "epoch": epoch,
            "nets": {key: item.to_checkpoint() for key, item in self.nets.items()},
            "optimizers": {key: item.state_dict() for key, item in self.optimizers.items()}
        }
        
        epoch_save = str(epoch).zfill(10)
        torch.save(
            save,
            path_model.format(epoch_save)
        )
        
        # Cleanup
        
        keep = self.measurements.best_models()
        for e in self.keep_models:
            if e not in keep:
                e = str(e).zfill(10)
                if os.path.isfile(path_model.format(e)):
                    os.remove(path_model.format(e))
        
        if epoch-1 not in keep:
            e = str(epoch-1).zfill(10)
            if os.path.isfile(path_model.format(e)):
                os.remove(path_model.format(e))
        
        self.keep_models = keep
        
        # Checkpoint measures
        
        m_i, m_other = self.measurements.get_measurements_i(epoch)
        
        torch.save(
            m_i,
            path_measures.format(epoch_save)
        )
        
        torch.save(
            m_other,
            path_measures.format("other")
        )
            
    ###########################################################################
    def load(self, path:str, device):
        import glob
        file = next(reversed(sorted(
            glob.glob(os.path.join(path, "*.model"))
        )))
        epoch = int(os.path.basename(file).split(".")[0])
        
        print(f"Checkpoint [{epoch}]: {file}")
        
        checkpoint = torch.load(file, map_location=device)
        
        for name, net_chkpt in checkpoint["nets"].items():
            self.nets[name].load(net_chkpt["state"])
            self.optimizers[name].load_state_dict(checkpoint["optimizers"][name])
        
        if self.measurements is not None:
            for e in range(epoch+1):
                m_i = torch.load(os.path.join(path, f"{str(e).zfill(10)}.measures"), map_location=device)
                
                self.measurements.fake_images.append(
                    m_i["GANMeasures.fake_images"]
                )
                self.measurements.fid_scores.append(
                    m_i["GANMeasures.fid_scores"]
                )
                self.measurements.average_profiles.append(
                    m_i["GANMeasures.average_profiles"]
                )
                if "GANMeasures.spectral_losses" in m_i:
                    self.measurements.spectral_losses.append(
                        m_i["GANMeasures.spectral_losses"]
                    )
                self.measurements.average_profile_losses_l1.append(
                    m_i["GANMeasures.average_profile_losses_l1"] 
                )
                
                del m_i
            
            m_other = torch.load(os.path.join(path, "other.measures"), map_location=device)
            for name, val in m_other.items():
                name_attr = name.replace("GANMeasures.", "")
                setattr(self.measurements, name_attr, val)
            
            self.measurements.fixed_noise = m_other["GANMeasures.fixed_noise"].to(device)
            
            print("Measures loaded from checkpoint.")
            del m_other
        else:
            print("WARNING: No measures loaded.")
        
        del checkpoint
        
        return epoch
    
    ###########################################################################
    def load_g(self, epoch:int, device):

        checkpoint = torch.load(
            os.path.join(self.folder, str(epoch).zfill(10) + ".model"),
            map_location = "cpu"
        )
        
        return Generator.from_checkpoint(checkpoint["nets"]["netG"]).to(device)