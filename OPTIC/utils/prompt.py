import torch
import torch.nn as nn
import torch.nn.functional as F
from skimage import exposure
import skimage.feature as feature
import pandas as pd

class Global_Prompt(nn.Module):
    def __init__(self, prompt_alpha=0.01, image_size=512):
        super().__init__()
        self.prompt_size = int(image_size * prompt_alpha) if int(image_size * prompt_alpha) > 1 else 1
        self.padding_size = (image_size - self.prompt_size)//2
        self.init_para = torch.ones((1, 3, self.prompt_size, self.prompt_size))
        self.data_prompt = nn.Parameter(self.init_para, requires_grad=True)
        self.pre_prompt = self.data_prompt.detach().cpu().data

    def update(self, init_data):
        with torch.no_grad():
            self.data_prompt.copy_(init_data)

    def iFFT(self, amp_src_, pha_src, imgH, imgW):
        # recompose fft
        real = torch.cos(pha_src) * amp_src_
        imag = torch.sin(pha_src) * amp_src_
        fft_src_ = torch.complex(real=real, imag=imag)

        src_in_trg = torch.fft.ifft2(fft_src_, dim=(-2, -1), s=[imgH, imgW]).real
        return src_in_trg

    def forward(self, x):
        _, _, imgH, imgW = x.size()

        fft = torch.fft.fft2(x.clone(), dim=(-2, -1))

        # extract amplitude and phase of both ffts
        amp_src, pha_src = torch.abs(fft), torch.angle(fft)
        amp_src = torch.fft.fftshift(amp_src)

        # obtain the low frequency amplitude part
        prompt = F.pad(self.data_prompt, [self.padding_size, imgH - self.padding_size - self.prompt_size,
                                          self.padding_size, imgW - self.padding_size - self.prompt_size],
                       mode='constant', value=1.0).contiguous()

        amp_src_ = amp_src * prompt
        amp_src_ = torch.fft.ifftshift(amp_src_)

        amp_low_ = amp_src[:, :, self.padding_size:self.padding_size+self.prompt_size, self.padding_size:self.padding_size+self.prompt_size]

        src_in_trg = self.iFFT(amp_src_, pha_src, imgH, imgW)
        return src_in_trg, amp_low_


class Instance_Prompt(nn.Module):  
    def __init__(self, prompt_size=1, image_size=512, time=0, previous_prompt=None):
        super().__init__()
        self.prompt_size = prompt_size
        self.padding_size = (image_size - self.prompt_size)//2+1
        
        if self.prompt_size == 1:
            self.init_para = torch.ones((1, 3, self.prompt_size, self.prompt_size))
            self.data_prompt = nn.Parameter(self.init_para, requires_grad=True)
            self.pre_prompt = self.data_prompt.detach().cpu().data
        else:
            self.init_para = torch.ones((1, 3, self.prompt_size, self.prompt_size))
            self.init_para[:,:, 1: self.prompt_size-1, 1: self.prompt_size-1] = previous_prompt
            self.data_prompt = nn.Parameter(self.init_para, requires_grad=True)
            self.pre_prompt = self.data_prompt.detach().cpu().data
            # Freeze the center part
            self.freeze_center()
            # self.data_prompt_size1 = torch.ones((1, 3, self.prompt_size, self.prompt_size))
    
    def update(self, init_data):
        with torch.no_grad():
            self.data_prompt.copy_(init_data)
    
    def freeze_center(self):
        """Freeze the center part of the prompt."""
        with torch.no_grad():
            center_region = torch.zeros_like(self.data_prompt)
            center_region[:, :, 1:self.prompt_size-1, 1:self.prompt_size-1] = 1

        def hook_fn(grad):
            # Zero out gradients in the center region
            grad[:, :, 1:self.prompt_size-1, 1:self.prompt_size-1] = 0
            return grad

        self.data_prompt.register_hook(hook_fn)
        
    def iFFT(self, amp_src_, pha_src, imgH, imgW):
        # recompose fft
        real = torch.cos(pha_src) * amp_src_
        imag = torch.sin(pha_src) * amp_src_
        fft_src_ = torch.complex(real=real, imag=imag)

        src_in_trg = torch.fft.ifft2(fft_src_, dim=(-2, -1), s=[imgH, imgW]).real
        return src_in_trg

    def forward(self, x):
        _, _, imgH, imgW = x.size()

        fft = torch.fft.fft2(x.clone(), dim=(-2, -1))

        # extract amplitude and phase of both ffts
        amp_src, pha_src = torch.abs(fft), torch.angle(fft)
        amp_src = torch.fft.fftshift(amp_src)
        
        # obtain the low frequency amplitude part
        prompt = F.pad(self.data_prompt, [self.padding_size, imgH - self.padding_size - self.prompt_size,
                                          self.padding_size, imgW - self.padding_size - self.prompt_size],
                       mode='constant', value=1.0).contiguous()

        amp_src_ = amp_src * prompt
        amp_src_ = torch.fft.ifftshift(amp_src_)

        amp_low_ = amp_src[:, :, self.padding_size:self.padding_size+self.prompt_size, self.padding_size:self.padding_size+self.prompt_size]

        src_in_trg = self.iFFT(amp_src_, pha_src, imgH, imgW)
        return src_in_trg, amp_low_


 