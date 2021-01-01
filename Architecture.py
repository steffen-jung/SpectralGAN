# -*- coding: utf-8 -*-
"""
Created on Sun Jun 28 23:00:08 2020

@author: Steff
"""

import torch
import torch.nn as nn
import numpy as np
from SpectralLoss import SpectralLoss

##############################################################################
#
#                              Generator Code
#
##############################################################################

class Generator(nn.Module):
    
    ###########################################################################
    def __init__(
            self,
            img_size            = 64,
            nz                  = 128,
            ngf                 = 32,
            kg                  = 4,
            nc                  = 3,
            last_layer_large    = True,
            last_layer_upsample = False,
            last_layer_stacked  = True
        ):
        
        super(Generator, self).__init__()
        
        self.img_size = img_size
        self.nz = nz
        self.ngf = ngf
        self.kg = kg
        self.nc = nc
        self.last_layer_large = last_layer_large
        self.last_layer_upsample = last_layer_upsample
        self.last_layer_stacked = last_layer_stacked
        
        transpose_layers = int(np.log2(img_size)) - 3
        
        if last_layer_upsample:
            ngf_upconv = ngf * (2**transpose_layers)
        else:
            ngf_upconv = ngf * (2**(transpose_layers+1))
        
        layers = nn.Sequential()
        layers.add_module("ConvTranspose1", nn.ConvTranspose2d(nz, min(512, ngf_upconv), kg, 1, 0, bias=False))
        layers.add_module("BatchNorm1",     nn.BatchNorm2d(min(512, ngf_upconv)))
        layers.add_module("ReLU1",          nn.ReLU())
        
        self._add_upconvs(layers, transpose_layers, ngf_upconv, kg)

        if last_layer_upsample:
            layers.add_module("Upsample", nn.Upsample(scale_factor=2, mode="nearest"))
        else:
            if last_layer_large:
                ksize = 8
                stride = 2
                padding = 3
            else:
                ksize = kg
                stride = 2
                padding = 1
            if last_layer_stacked:  channels = ngf
            else:                   channels = 3
            layers.add_module(f"ConvTranspose{transpose_layers+2}", nn.ConvTranspose2d(ngf * 2, channels, ksize, stride, padding, bias=False))
            if last_layer_stacked:
                layers.add_module(f"ReLU{transpose_layers+2}", nn.ReLU())
            
        if last_layer_stacked:
            layers.add_module("Conv1", nn.Conv2d(ngf, ngf, 5, padding=2)
            )
            layers.add_module(f"ReLU{transpose_layers+3}", nn.ReLU())
            layers.add_module("Conv2",                     nn.Conv2d(ngf, ngf, 5, padding=2))
            layers.add_module(f"ReLU{transpose_layers+4}", nn.ReLU())
            layers.add_module("Conv3",                     nn.Conv2d(ngf,  nc, 5, padding=2))
        
        layers.add_module("Tanh", nn.Tanh())

        self._forward = layers
        
    ###########################################################################
    def forward(self, x):
        x = self._forward(x)
        return x
    
    ###########################################################################
    def _add_upconvs(self, layers, n, d, kg):
        for i in range(n):
            layers.add_module(
                f"ConvTranspose{i+2}",
                nn.ConvTranspose2d(
                    in_channels  = min(512, d),
                    out_channels = min(512, d // 2),
                    kernel_size  = kg,
                    stride       = 2,
                    padding      = 1,
                    bias         = False
                )
            )
            layers.add_module(f"BatchNorm{i+2}", nn.BatchNorm2d(min(512, d // 2)))
            layers.add_module(f"ReLU{i+2}",      nn.ReLU())
            d = d // 2
            
    ###########################################################################
    def par_count(self):
        c = 0
        for p in self.parameters():
            c += np.prod(p.shape)
        return c
    
    ###########################################################################
    def print_par_count(self):
        for name, p in self.named_parameters():
            print(f"{name:>40}: {str(p.shape):>40} {np.prod(p.shape):>15,}")
            
    ###########################################################################
    def load(self, state):
        self.load_state_dict(state)
        return self
            
    ###########################################################################
    def to_checkpoint(self):
        chkpt = {}
        chkpt["state"]   = self.state_dict()
        chkpt["pars"] = {
            "img_size"            : self.img_size,
            "nz"                  : self.nz,
            "ngf"                 : self.ngf,
            "kg"                  : self.kg,
            "nc"                  : self.nc,
            "last_layer_large"    : self.last_layer_large,
            "last_layer_upsample" : self.last_layer_upsample,
            "last_layer_stacked"  : self.last_layer_stacked
        }
        return chkpt

    ###########################################################################
    @staticmethod
    def from_checkpoint(chkpt):
        G = Generator(**chkpt["pars"])
        G.load(chkpt["state"])
        return G

##############################################################################
#
#                              Discriminator Code
#
##############################################################################

class Discriminator(nn.Module):
    
    ###########################################################################
    def __init__(
            self,
            img_size   :  int = 64,
            ndf        :  int = 64,
            kd         :  int = 4,
            nc         :  int = 3,
            batch_norm : bool = True
        ):
        
        super(Discriminator, self).__init__()
        
        self.img_size = img_size
        self.ndf = ndf
        self.kd = kd
        self.nc = nc
        
        # padding discriminator
        pd = 1
        # stride discriminator
        sd = 2
        
        self.spectral_transform = SpectralLoss(rows=img_size, cols=img_size)
        # self.register_buffer("spectral_transform", SpectralLoss(rows=img_size, cols=img_size))
        
        layers = nn.Sequential()
        layers.add_module("Conv1",                    nn.Conv2d(nc, ndf, kd, sd, pd, bias=False))
        layers.add_module("ReLU1",                    nn.LeakyReLU(0.2))

        blocks = int(np.log2(img_size))-3
        for i in range(blocks):
            f_in  = ndf * (2 ** i)
            f_in  = min(f_in, 512)
            f_out = ndf * (2 ** (i+1))
            f_out  = min(f_out, 512)
            layers.add_module(f"Conv{2+i}",           nn.Conv2d(f_in, f_out, kd, sd, pd, bias=False))
            if batch_norm:
                layers.add_module(f"BatchNorm{2+i}",  nn.BatchNorm2d(f_out))
            layers.add_module(f"ReLU{2+i}",           nn.LeakyReLU(0.2))
            
        f_in = min(ndf * (2** blocks), 512)
        layers.add_module(f"Conv{2+blocks}",          nn.Conv2d(f_in, 1, kd, 1, 0, bias=False))
        
        self._forward = layers
        
    ###########################################################################
    def forward(self, x):
        y = self._forward(x)
        
        return y
    
    ###########################################################################
    def par_count(self):
        c = 0
        for p in self.parameters():
            c += np.prod(p.shape)
        return c
    
    ###########################################################################
    def print_par_count(self):
        for name, p in self.named_parameters():
            print(f"{name:>40}: {str(p.shape):>40} {np.prod(p.shape):>15,}")
    
    ###########################################################################
    def load(self, state):
        self.load_state_dict(state)
            
    ###########################################################################
    def to_checkpoint(self):
        chkpt = {}
        chkpt["state"]   = self.state_dict()
        chkpt["pars"] = {
            "img_size"   : self.img_size,
            "ndf"        : self.ndf,
            "kd"         : self.kd,
            "nc"         : self.nc
        }
        return chkpt

    ###########################################################################
    @staticmethod
    def from_checkpoint(chkpt):
        D = Discriminator(**chkpt["pars"])
        D.load(chkpt["state"])
        return D
    
##############################################################################

class Unnormalize(nn.Module):
    
    ###########################################################################
    def __init__(self):
        super(Unnormalize, self).__init__()
        
    ###########################################################################
    def forward(self, input):
        return (input + 1) / 2
    
##############################################################################

class Normalize(nn.Module):
    
    ###########################################################################
    def __init__(self):
        super(Normalize, self).__init__()
        
    ###########################################################################
    def forward(self, input):
        return (input - 0.5) / 0.5
    
##############################################################################
#
#                              Spectral Discriminator Code
#
##############################################################################

class SpectralDiscriminator(nn.Module):
    
    ###########################################################################
    def __init__(self, img_size = 64, spectral = "linear"):
        super(SpectralDiscriminator, self).__init__()
        
        self.img_size = img_size
        self.spectral = spectral

        self.spectral_transform = SpectralLoss(rows=img_size, cols=img_size)

        self._add_spectral_layers(spectral)
        
    ###########################################################################
    def _add_spectral_layers(self, spectral):
        if spectral == "none":
            self.forward = self.forward_none

        else:
            layers = nn.Sequential()
            
            if "unnormalize" in spectral:
                layers.add_module("Unnormalize", Unnormalize())
                
            if "dropout" in spectral:
                layers.add_module("Dropout", nn.Dropout())
                
            if "linear" in spectral and not "nonlinear" in spectral:
                layers.add_module("LinearSpectral", nn.Linear(self.spectral_transform.vector_length, 1))
    
            if "nonlinear" in spectral:
                layers.add_module("Linear1Spectral", nn.Linear(self.spectral_transform.vector_length, self.spectral_transform.vector_length))
                layers.add_module("ReLU1Spectral",   nn.LeakyReLU(0.2))
                layers.add_module("Linear2Spectral", nn.Linear(self.spectral_transform.vector_length, 1))
                
            self._forward_spectral = layers
            
    ###########################################################################
    def forward(self, x):
        x_profiles = self.spectral_transform.spectral_vector(x)
        y = self._forward_spectral(x_profiles)
        
        return y
    
    ###########################################################################
    def forward_none(self, x):
        return torch.tensor(0.0)
    
    ###########################################################################
    def par_count(self):
        c = 0
        for p in self.parameters():
            c += np.prod(p.shape)
        return c
    
    ###########################################################################
    def print_par_count(self):
        for name, p in self.named_parameters():
            print(f"{name:>40}: {str(p.shape):>40} {np.prod(p.shape):>15,}")
    
    ###########################################################################
    def load(self, state):
        self.load_state_dict(state)
            
    ###########################################################################
    def to_checkpoint(self):
        chkpt = {}
        chkpt["state"]   = self.state_dict()
        chkpt["pars"] = {
            "img_size"   : self.img_size,
            "spectral"   : self.spectral,
        }
        return chkpt

    ###########################################################################
    @staticmethod
    def from_checkpoint(chkpt):
        D = Discriminator(**chkpt["pars"])
        D.load(chkpt["state"])
        return D
    
##############################################################################
#
#                       Spectral Discriminator 2D Code
#
##############################################################################

class SpectralDiscriminator2D(nn.Module):
    
    ###########################################################################
    def __init__(self, img_size = 64, spectral = "linear"):
        super(SpectralDiscriminator, self).__init__()
        
        self.img_size = img_size
        self.spectral = spectral

        self.spectral_transform = SpectralLoss(rows=img_size, cols=img_size)

        self._add_spectral_layers(spectral)
        
    ###########################################################################
    def _add_spectral_layers(self, spectral):
        if spectral == "none":
            self.forward = self.forward_none

        else:
            layers = nn.Sequential()
            
            if "unnormalize" in spectral:
                layers.add_module("Unnormalize", Unnormalize())
                
            if "dropout" in spectral:
                layers.add_module("Dropout", nn.Dropout())
                
            if "linear" in spectral and not "nonlinear" in spectral:
                layers.add_module("LinearSpectral", nn.Linear(self.spectral_transform.vector_length, 1))
    
            if "nonlinear" in spectral:
                layers.add_module("Linear1Spectral", nn.Linear(self.spectral_transform.vector_length, self.spectral_transform.vector_length))
                layers.add_module("ReLU1Spectral",   nn.LeakyReLU(0.2))
                layers.add_module("Linear2Spectral", nn.Linear(self.spectral_transform.vector_length, 1))
                
            self._forward_spectral = layers
            
    ###########################################################################
    def forward(self, x):
        x_profiles = self.spectral_transform.spectral_vector(x)
        y = self._forward_spectral(x_profiles)
        
        return y
    
    ###########################################################################
    def forward_none(self, x):
        return torch.tensor(0.0)
    
    ###########################################################################
    def par_count(self):
        c = 0
        for p in self.parameters():
            c += np.prod(p.shape)
        return c
    
    ###########################################################################
    def print_par_count(self):
        for name, p in self.named_parameters():
            print(f"{name:>40}: {str(p.shape):>40} {np.prod(p.shape):>15,}")
    
    ###########################################################################
    def load(self, state):
        self.load_state_dict(state)
            
    ###########################################################################
    def to_checkpoint(self):
        chkpt = {}
        chkpt["state"]   = self.state_dict()
        chkpt["pars"] = {
            "img_size"   : self.img_size,
            "spectral"   : self.spectral,
        }
        return chkpt

    ###########################################################################
    @staticmethod
    def from_checkpoint(chkpt):
        D = Discriminator(**chkpt["pars"])
        D.load(chkpt["state"])
        return D
    
##############################################################################
#
#                              Loss Code
#
##############################################################################

class Loss(nn.Module):
    
    ###########################################################################
    def __init__(
            self,
            loss             : str = "gan",
            gradient_penalty : str = "none",
            device           : str = "cpu"
        ):
        
        super(Loss, self).__init__()
        
        self.device = device
        self.zero = torch.tensor(0.0)
        
        self.loss = loss
        self.gradient_penalty = gradient_penalty
        
        # set up loss function
        
        self.criterion = None
        
        if   loss == "gan":
            self.criterion = nn.BCEWithLogitsLoss()
            self._loss = self._gan
            
        elif loss == "lsgan":
            self.criterion = nn.MSELoss()
            self._loss = self._gan
            
        elif loss == "wgan":
            self._loss = self._wgan

        else:
            raise NotImplementedError(f"Loss not implemented: {loss}")
            
        # set up gradient penalty
        
        if   gradient_penalty == "none":
            self._gp = self._non_gp
            
        elif gradient_penalty == "wgan-gp":
            self._gp = self._wgan_gp
        
        else:
            raise NotImplementedError(f"Gradient penalty not implemented: {gradient_penalty}")
            
        # set up weight clipping
        
        if loss not in ["wgan"] or gradient_penalty in ["wgan-gp"]:
            self._weight_clipping = self._no_weight_clipping
    
    ###########################################################################
    def _gan(self, y:torch.tensor, target:bool, **kwargs) -> torch.tensor:
        batch_size = y.size(0)
        label = torch.full(
            (batch_size,),
            target,
            dtype = torch.float,
            device = self.device
        )
        return self.criterion(y.view(-1), label)
    
    ###########################################################################
    def _wgan(self, y:torch.tensor, target:bool, **kwargs) -> torch.tensor:
        if target:
            return -y.mean()
        return y.mean()
    
    ###########################################################################
    def _weight_clipping(self, net:torch.nn.Module, **kwargs):
        for p in net.parameters():
            p.requires_grad = False
            p.clamp_(
                min = -0.01,
                max =  0.01
            )
            p.requires_grad = True
            
    ###########################################################################
    def _no_weight_clipping(self, net:torch.nn.Module, **kwargs):
        pass
    
    ###########################################################################
    def _non_gp(self, **kwargs) -> torch.tensor:
        return self.zero
    
    ###########################################################################
    def _wgan_gp(self, reals:torch.tensor, fakes:torch.tensor, D:nn.Module=None, lambd:float=10.0, **kwargs) -> torch.tensor:
        assert reals.shape == fakes.shape
        
        batch_size = reals.size(0)
        channel_size = reals.size(1)
        w = reals.size(2)
        h = reals.size(3)
        
        eps = np.random.rand(batch_size)
        eps = torch.from_numpy(eps).to(self.device)
        eps = eps.expand(channel_size, w, h, batch_size)
        eps = eps.permute(3,0,1,2)
        
        x = eps * reals + (1-eps) * fakes.detach()
        x = x.float()
        x.requires_grad = True
        y = D(x)
        
        grad_outputs = torch.ones_like(y)
        grad = torch.autograd.grad(
            outputs = y,
            inputs = x,
            grad_outputs = grad_outputs,
            retain_graph = True,
            create_graph = True
        )
        grad = grad[0].view(batch_size,-1)
        
        #penalty = (torch.norm(grad[0].view(batch_size,-1), dim=1) - 1).pow(2)
        penalty = (torch.sqrt(torch.sum(grad**2, dim=1) + 1E-8) - 1.0)**2
        penalty = lambd * penalty
        
        return penalty.mean()
    
    ###########################################################################
    def __str__(self):
        return f"Criterion: {self.criterion}\nLoss: {self._loss}\nGradient Penalty: {self._gp}\nWeight Clipping: {self._weight_clipping}"
    
    def __repr__(self):
        return "Loss"
    
    ###########################################################################
    # def forward(self, y:torch.tensor, target:bool) -> torch.tensor:
    #     return self._loss(y, target) + self._gp()
    
##############################################################################
#
#                              Optimizer Code
#
##############################################################################

class Optimizer:
    
    ###########################################################################
    def __init__(self, net:torch.nn.Module, loss:str="gan", gradient_penalty:str="none"):
        if loss in ["gan", "lsgan"]:
            self.optimizer = torch.optim.Adam(
                net.parameters(),
                lr = 2E-4,
                betas = (0.5, 0.999)
            )
            
        elif loss in ["wgan"]:
            if gradient_penalty in ["wgan-gp"]:
                self.optimizer = torch.optim.Adam(
                    net.parameters(),
                    lr = 1E-4,
                    betas = (0.0, 0.9)
                )
            else:
                self.optimizer = torch.optim.RMSprop(
                    net.parameters(),
                    lr = 5E-5
                )
    
##############################################################################
#
#                              Debugging
#
##############################################################################
    
if __name__ == "__main__":
    # print(str(Generator(  64)))
    # print(str(Generator( 128)))
    # print(str(Generator( 256)))
    # print(str(Generator( 512)))
    # print(str(Generator(1024)))
    
    # m = Discriminator(64, spectral="linear", combine="mean")
    # print(str(m))
    # print(f"{m.par_count():,}")
    # m.print_par_count()
    
    # m = Generator(64)
    # print(str(m))
    # print(f"{m.par_count():,}")
    # m.print_par_count()
    
    # m = np.array([1/2,1/3])
    # imgs1 = np.zeros((3,10,10))
    # imgs1.fill(2)
    # imgs2 = np.zeros((3,10,10))
    # imgs2.fill(3)
    # imgs = np.stack((imgs1,imgs2))
    # imgs = torch.from_numpy(imgs)
    # m = torch.from_numpy(m).expand(imgs.size(1),imgs.size(2),imgs.size(3),imgs.size(0)).permute(3,0,1,2)
    
    import torch
    from torch import nn as nn
    test = nn.Sequential()
    test.add_module("L1",nn.Linear(10,10))
    test.add_module("L2",nn.Linear(10,5))
    test.add_module("ReLU",nn.ReLU())
    test.add_module("L3",nn.Linear(5,3))
    
    # y = test(x)
    # y.sum().backward()
    #print(x.grad)
        
    # y = test(x)
    # y.mean().backward()
    # print(x.grad)
        
    # x = torch.randn(5,10)
    # x.requires_grad = True
    
    x = torch.randn(5,10)
    x.requires_grad = True
    y = test(x)
    print(y)
    grad_outputs = torch.ones_like(y)
    grad = torch.autograd.grad(outputs=y, inputs=x, grad_outputs=grad_outputs)
    print(grad)
    
    x = torch.randn(5,10)
    x.requires_grad = True
    y = test(x)
    print(y)
    grad_outputs = torch.ones_like(y)
    grad = torch.autograd.grad(outputs=y, inputs=x, grad_outputs=grad_outputs)
    print(grad)
    
    # for name, p in test.named_parameters():
    #     print(p.grad)
    
    print("loss impl.")
    f = test
    
    x = torch.randn(5,10) # _interpolate(real, fake).detach()
    x.requires_grad = True
    pred = f(x)
    grad = torch.autograd.grad(pred, x, grad_outputs=torch.ones_like(pred), create_graph=True)
    #norm = grad.view(grad.size(0), -1).norm(p=p_norm, dim=1)
    print(grad)
    
    
    class Testtest:
        
        def __init__(self):
            self.a = self.b
        
        def a(self):
            pass
        
        def b(self):
            pass
        
        def __str__(self):
            return f"a: {self.a}"
        
        def __repr__(self):
            return "Loss"
        
    test = Testtest()