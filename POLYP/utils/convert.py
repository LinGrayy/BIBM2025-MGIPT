import torch.nn as nn
import numpy as np


class AdaBN(nn.BatchNorm2d):
    def __init__(self, in_ch, warm_n=5, lambda_s=0.8):
        super(AdaBN, self).__init__(in_ch)
        self.warm_n = warm_n
        self.sample_num = 0
        self.new_sample = False
        self.moment = lambda_s
        self.inference = False

    def get_mu_var(self, x):
        if self.new_sample:
            self.sample_num += 1
        C = x.shape[1]

        cur_mu = x.mean((0, 2, 3), keepdims=True).detach()
        cur_var = x.var((0, 2, 3), keepdims=True).detach()

        src_mu = self.running_mean.view(1, C, 1, 1)
        src_var = self.running_var.view(1, C, 1, 1)

        moment = 1 / ((np.sqrt(self.sample_num) / self.warm_n) + 1)

        new_mu = moment * cur_mu + (1 - moment) * src_mu
        new_var = moment * cur_var + (1 - moment) * src_var
        return new_mu, new_var


    def forward(self, x): # no combine in training, sita in inference                                 #adabn is replace source bn parameter
        N, C, H, W = x.shape

        src_mu = self.running_mean.view(1, C, 1, 1)
        src_var = self.running_var.view(1, C, 1, 1)
        
        cur_mu = x.mean((2, 3), keepdims=True) 
        cur_std = x.std((2, 3), keepdims=True) 
        self.src_mu = src_mu
        self.src_var = src_var
        self.cur_mu = cur_mu
        self.cur_std =cur_std

        self.bn_loss = (
                (src_mu - cur_mu).abs().mean() + (src_var.sqrt() - cur_std).abs().mean()
        )

        # Normalization with new statistics
        new_sig = (cur_std * cur_std + self.eps).sqrt()
        new_x = ((x - cur_mu) / new_sig) * self.weight.view(1, C, 1, 1) + self.bias.view(1, C, 1, 1)
        self.new_x = new_x
        # inference 
        moment = self.moment
        self.new_mu_inference = moment * src_mu + (1 - moment) * cur_mu
        self.new_var_inference = moment * src_var + (1 - moment) * x.var((0, 2, 3), keepdims=True)
        new_sig_inference = (self.new_var_inference + self.eps).sqrt()
        self.new_x_inference = ((x - self.new_mu_inference) / new_sig_inference) * self.weight.view(1, C, 1, 1) + self.bias.view(1, C, 1, 1)
        if self.inference: 
            self.new_x = self.new_x_inference
        
        return self.new_x
    
