import os
import torch
import numpy as np
import argparse, sys, datetime
from config import Logger
from torch.autograd import Variable
from utils.convert import AdaBN
from utils.memory import Memory
from utils.prompt import Prompt_MultiResolution2_Uncenter as Prompt
from utils.prompt import Prompt as VPTTA_Prompt
from utils.metrics import calculate_metrics
from networks.PraNet_Res2Net_TTA import PraNet
from torch.utils.data import DataLoader
from dataloaders.POLYP_dataloader import POLYP_dataset
from dataloaders.convert_csv_to_list import convert_labeled_list
from PIL import Image
import torchvision.transforms as transforms
import torch.nn.functional as F

torch.set_num_threads(1)
import matplotlib.pyplot as plt
import numpy as np
import random

def dice_loss(pred, target, smooth=1e-5):
    pred = pred.contiguous().view(-1) 
    target = target.contiguous().view(-1)  

    intersection = (pred * target).sum()

    dice = (2. * intersection + smooth) / (pred.sum() + target.sum() + smooth)
    
    return 1 - dice


class MGIPT: # no MB, with EMA teacher global prompt
    def __init__(self, config):
        # Save Log
        time_now = datetime.datetime.now().__format__("%Y%m%d_%H%M%S_%f")
        log_root = os.path.join(config.path_save_log, 'ours')
        if not os.path.exists(log_root):
            os.makedirs(log_root)
        log_path = os.path.join(log_root, time_now + '.log')
        sys.stdout = Logger(log_path, sys.stdout)
        self.config = config
        # Data Loading
        target_test_csv = []
        for target in config.Target_Dataset:
            target_test_csv.append(target + '_train.csv')
            target_test_csv.append(target + '_test.csv')
        ts_img_list, ts_label_list = convert_labeled_list(config.dataset_root, target_test_csv)
        target_test_dataset = POLYP_dataset(config.dataset_root, ts_img_list, ts_label_list,
                                            config.image_size)
        self.target_test_loader = DataLoader(dataset=target_test_dataset,
                                             batch_size=config.batch_size,
                                             shuffle=False,
                                             pin_memory=True,
                                             drop_last=False,
                                             num_workers=config.num_workers)
        self.image_size = config.image_size

        # Model
        self.load_model = os.path.join(config.model_root, str(config.Source_Dataset))  # Pre-trained Source Model
        self.in_ch = config.in_ch
        self.out_ch = config.out_ch

        # Optimizer
        self.optim = config.optimizer
        self.lr = config.lr
        self.weight_decay = config.weight_decay
        self.momentum = config.momentum
        self.betas = (config.beta1, config.beta2)

        # GPU
        self.device = config.device

        # Prompt
        self.prompt_alpha = config.prompt_alpha
        self.iters = config.iters

        # Initialize the pre-trained model and optimizer
        self.build_model()

        self.previous_prompt = None
        self.tmp_pred_logit = None
        # Print Information
        for arg, value in vars(config).items():
            print(f"{arg}: {value}")
        self.print_prompt()
        print('***' * 20)

    def build_model(self):
        self.prompt = Prompt(prompt_size=1, image_size=self.image_size).to(self.device)
        self.model = PraNet().to(self.device)
        checkpoint = torch.load(os.path.join(self.load_model, 'pretrain-PraNet.pth'))
        self.model.load_state_dict(checkpoint, strict=True)

        if self.optim == 'SGD':
            self.optimizer = torch.optim.SGD(
                self.prompt.parameters(),
                lr=self.lr,
                momentum=self.momentum,
                nesterov=True,
                weight_decay=self.weight_decay
            )
        elif self.optim == 'Adam':
            self.optimizer = torch.optim.Adam(
                self.prompt.parameters(),
                lr=self.lr,
                betas=self.betas,
                weight_decay=self.weight_decay
            )

        self.augmentations_list = [
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            # transforms.ColorJitter(brightness=0.5),
        ]
 
    def print_prompt(self):
        num_params = 0
        for p in self.prompt.parameters():
            num_params += p.numel()
        print("The number of total parameters: {}".format(num_params))
            
    def apply_augmentations(self, x, augmentation):
        augmented_imgs = []
        for i in range(x.size(0)):  
            img = x[i].cpu().numpy().transpose(1, 2, 0)  
            img_pil = Image.fromarray((img * 255).astype('uint8')) 
            img_aug = augmentation(img_pil) 
            img_aug_tensor = transforms.ToTensor()(img_aug).to(self.device)  
            augmented_imgs.append(img_aug_tensor)

        return torch.stack(augmented_imgs)  
    
    def run(self): 
        metric_dict = ['Dice', 'Enhanced_Align', 'Structure_Measure']

        metrics_test = [[], [], []]

        # Valid on Target
        epochs_prompt_size_1 = 6 # Number of epochs for each prompt size
        loss_values=[]

        self.global_prompt = VPTTA_Prompt(prompt_alpha=self.prompt_alpha, image_size=self.image_size).to(self.device)
        self.global_prompt1 = VPTTA_Prompt(prompt_alpha=0.001, image_size=self.image_size).to(self.device)
        self.global_prompt2 = VPTTA_Prompt(prompt_alpha=0.006, image_size=self.image_size).to(self.device)
        
        self.global_prompt_teacher = VPTTA_Prompt(prompt_alpha=self.prompt_alpha, image_size=self.image_size).to(self.device)
        self.global_prompt_teacher1 = VPTTA_Prompt(prompt_alpha=0.001, image_size=self.image_size).to(self.device)
        self.global_prompt_teacher2 = VPTTA_Prompt(prompt_alpha=0.006, image_size=self.image_size).to(self.device)
           
        for batch, data in enumerate(self.target_test_loader):
            x, y, path = data
            x, y = Variable(x).to(self.device), Variable(y).to(self.device)

            self.model.eval()
            self.prompt.train()
            self.model.change_BN_status(new_sample=True)

            # Initialize Prompt
            init_data0 = torch.ones((1, 3, self.prompt.prompt_size, self.prompt.prompt_size)).data
            self.prompt.update(init_data0)
            self.prompt_level2 = Prompt(prompt_size=3, image_size=self.image_size, previous_prompt=self.prompt.data_prompt.detach().cpu()).to(self.device)
            self.prompt_level3 = VPTTA_Prompt(prompt_alpha=self.prompt_alpha, image_size=self.image_size).to(self.device)
            
            MV_loss_list = []
            # Prompt Training
            for augmentation in self.augmentations_list:
                x_augmented = self.apply_augmentations(x, augmentation)
                # train global prompt
                prompts = [self.global_prompt, self.global_prompt1, self.global_prompt2]
                for prompt in prompts:
                    self.optimizer = torch.optim.Adam(
                        list(prompt.parameters()),
                        lr=self.lr,
                        betas=self.betas,
                        weight_decay=self.weight_decay
                    )
                
                    g_prompt_x, _ = prompt(x)
                    self.model(g_prompt_x)
                    
                    # Calculate loss and backpropagate
                    times, bn_loss = 0, 0
                    for nm, m in self.model.named_modules():
                        if isinstance(m, AdaBN):
                            bn_loss += m.bn_loss
                            times += 1
                    loss = bn_loss / times 
                   
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()
                    self.model.change_BN_status(new_sample=False)
                
            # Train Prompt for n iters 
                for tr_iter in range(self.iters):
                    # Set optimizer                  
                    if tr_iter == 0:
                        if self.optim == 'SGD':
                                self.optimizer = torch.optim.SGD(
                                    list(self.prompt.parameters()),
                                    lr=self.lr,
                                    momentum=self.momentum,
                                    nesterov=True,
                                    weight_decay=self.weight_decay
                                )
                        elif self.optim == 'Adam':
                                self.optimizer = torch.optim.Adam(
                                    list(self.prompt.parameters()),
                                    lr=self.lr,
                                    betas=self.betas,
                                    weight_decay=self.weight_decay
                                )
                        for epoch in range(epochs_prompt_size_1):
                            prompt_x_aug, _ = self.prompt(x_augmented)                            
                            prompt_x, _ = self.prompt(x)
                            
                            pred_logit_orig = self.model(prompt_x)
                            
                            # Calculate loss and backpropagate
                            times, bn_loss = 0, 0
                            for nm, m in self.model.named_modules():
                                if isinstance(m, AdaBN):
                                    bn_loss += m.bn_loss
                                    times += 1
                            loss = bn_loss / times 
                            if epoch == 0:
                                save_bn_loss_first  = bn_loss
                            
                            # print(tr_iter, epoch, loss)
                            self.optimizer.zero_grad()
                            loss.backward()
                            self.optimizer.step()
                            self.model.change_BN_status(new_sample=False)
                            loss_values.append(loss.item())

                        
                        self.previous_prompt = self.prompt.data_prompt.detach().cpu()
                        prompt_x_aug, _ = self.prompt(x_augmented)                            
                        prompt_x, _ = self.prompt(x) 
                        pred_logit_aug = self.model(prompt_x_aug)
                        pred_logit_orig = self.model(prompt_x)
                        pseudo_labels = (torch.sigmoid(pred_logit_orig) > 0.5).float() 
                        pseudo_labels_aug = (torch.sigmoid(pred_logit_aug) > 0.5).float()  # Binarize the output to create pseudo labels
                        MV_loss_list.append(dice_loss(pseudo_labels, pseudo_labels_aug))
                        
                        
                    elif tr_iter ==1 : 
                        self.prompt_level2 = Prompt(prompt_size=3, image_size=self.image_size, previous_prompt=self.previous_prompt).to(self.device)
                        
                        if self.optim == 'SGD':
                            self.optimizer = torch.optim.SGD(
                                list(self.prompt_level2.parameters()), 
                                lr=self.lr,
                                momentum=self.momentum,
                                nesterov=True,
                                weight_decay=self.weight_decay
                            )
                        elif self.optim == 'Adam':
                            self.optimizer = torch.optim.Adam(
                                list(self.prompt_level2.parameters()),
                                lr=self.lr,
                                betas=self.betas,
                                weight_decay=self.weight_decay
                            )
                        for epoch in range(epochs_prompt_size_1):
                            prompt_x_aug, _ = self.prompt_level2(x_augmented)
                            prompt_x, _ = self.prompt_level2(x)
                            
                            pred_logit_orig = self.model(prompt_x)
                            
                            times, bn_loss = 0, 0
                            for nm, m in self.model.named_modules():
                                if isinstance(m, AdaBN):
                                    bn_loss += m.bn_loss
                                    times += 1
                            loss = bn_loss / times #+ pseudo_loss
                            self.optimizer.zero_grad()
                            loss.backward()
                            self.optimizer.step()
                            self.model.change_BN_status(new_sample=False)
                        
                        self.previous_prompt = self.prompt_level2.data_prompt.detach().cpu()
                        prompt_x_aug, _ = self.prompt_level2(x_augmented)
                        prompt_x, _ = self.prompt_level2(x)
                        pred_logit_aug= self.model(prompt_x_aug)
                        pred_logit_orig = self.model(prompt_x)
                        pseudo_labels = (torch.sigmoid(pred_logit_orig) > 0.5).float() 
                        pseudo_labels_aug = (torch.sigmoid(pred_logit_aug) > 0.5).float()  # Binarize the output to create pseudo labels
                        MV_loss_list.append(dice_loss(pseudo_labels, pseudo_labels_aug))
                        if MV_loss_list[0] < MV_loss_list[1]:
                            self.prompt_level3 = self.prompt
                            break 
                        
                    else: 
                        self.prompt_level3 = Prompt(prompt_size=5, image_size=self.image_size, previous_prompt=self.previous_prompt).to(self.device)
                        
                        if self.optim == 'SGD':
                            self.optimizer = torch.optim.SGD(
                                list(self.prompt_level3.parameters()), 
                                lr=self.lr,
                                momentum=self.momentum,
                                nesterov=True,
                                weight_decay=self.weight_decay
                            )
                        elif self.optim == 'Adam':
                            self.optimizer = torch.optim.Adam(
                                list(self.prompt_level3.parameters()), 
                                lr=self.lr,
                                betas=self.betas,
                                weight_decay=self.weight_decay
                            )
                        for epoch in range(epochs_prompt_size_1):
                            prompt_x_aug, _ = self.prompt_level3(x_augmented)
                            prompt_x, _ = self.prompt_level3(x)
                            pred_logit_orig = self.model(prompt_x)
                            
                            times, bn_loss = 0, 0
                            for nm, m in self.model.named_modules():
                                if isinstance(m, AdaBN):
                                    bn_loss += m.bn_loss
                                    times += 1
                            loss = bn_loss / times
                            self.optimizer.zero_grad()
                            loss.backward()
                            self.optimizer.step()
                            self.model.change_BN_status(new_sample=False)
                        
                        prompt_x_aug, _ = self.prompt_level3(x_augmented)
                        prompt_x, _ = self.prompt_level3(x)
                        pred_logit_aug = self.model(prompt_x_aug)
                        pred_logit_orig= self.model(prompt_x)
                        pseudo_labels = (torch.sigmoid(pred_logit_orig) > 0.5).float() 
                        pseudo_labels_aug = (torch.sigmoid(pred_logit_aug) > 0.5).float()  # Binarize the output to create pseudo labels
                        MV_loss_list.append(dice_loss(pseudo_labels, pseudo_labels_aug))
                        if MV_loss_list[1] < MV_loss_list[2]:
                            self.prompt_level3 = Prompt(prompt_size=5, image_size=self.image_size, previous_prompt=self.previous_prompt).to(self.device)            
            
            # EMA Teacher Prompt
            ema_decay = 0.1
            for teacher_param, student_param in zip(self.global_prompt_teacher.parameters(), self.global_prompt.parameters()):
                teacher_param.data.mul_(ema_decay).add_(student_param.data, alpha=1 - ema_decay)
            for teacher_param, student_param in zip(self.global_prompt_teacher1.parameters(), self.global_prompt1.parameters()):
                teacher_param.data.mul_(ema_decay).add_(student_param.data, alpha=1 - ema_decay)
            for teacher_param, student_param in zip(self.global_prompt_teacher2.parameters(), self.global_prompt2.parameters()):
                teacher_param.data.mul_(ema_decay).add_(student_param.data, alpha=1 - ema_decay)
           

            # Inference
            self.model.eval()
            self.prompt.eval()
            self.prompt_level2.eval()
            self.prompt_level3.eval()
            self.global_prompt_teacher.eval()
            self.global_prompt.eval()
            with torch.no_grad():
                prompt_x, _ = self.prompt_level3(x)
                g_prompt_x, low_freq = self.global_prompt_teacher(x)
                g_prompt_x1, low_freq1 = self.global_prompt_teacher1(x)
                g_prompt_x2, low_freq2 = self.global_prompt_teacher2(x)
                       
                for nm, m in self.model.named_modules():
                    if isinstance(m, AdaBN):
                        m.inference = True   
                pred_logit_o = self.model(prompt_x) 
                pred_logit_g = self.model(g_prompt_x) 
                pred_logit_g1 = self.model(g_prompt_x1) 
                pred_logit_g2 = self.model(g_prompt_x2) 
                
                conf_o = torch.softmax(pred_logit_o, dim=1).max(dim=1)[0]  
                conf_g = torch.softmax(pred_logit_g, dim=1).max(dim=1)[0]
                conf_g1 = torch.softmax(pred_logit_g1, dim=1).max(dim=1)[0]
                conf_g2 = torch.softmax(pred_logit_g2, dim=1).max(dim=1)[0]

                weight_o = conf_o / (conf_o + conf_g+ conf_g1+ conf_g2)*2
                weight_g = conf_g / (conf_o + conf_g+ conf_g1+ conf_g2)*0.5
                weight_g1 = conf_g1 / (conf_o + conf_g+ conf_g1+ conf_g2)*0.5
                weight_g2 = conf_g2 / (conf_o + conf_g+ conf_g1+ conf_g2)*0.5
                
                pred_logit = weight_o * pred_logit_o + weight_g * pred_logit_g+\
                    weight_g1 * pred_logit_g1+ weight_g2 * pred_logit_g2
           
            # Calculate the metrics
            seg_output = torch.sigmoid(pred_logit)
            metrics = calculate_metrics(seg_output.detach().cpu(), y.detach().cpu())
            
            for i in range(len(metrics)):
                assert isinstance(metrics[i], list), "The metrics value is not list type."
                metrics_test[i] += metrics[i]

        test_metrics_y = np.mean(metrics_test, axis=1)
        print_test_metric_mean = {}
        for i in range(len(test_metrics_y)):
            print_test_metric_mean[metric_dict[i]] = test_metrics_y[i]
        print("Test Metrics Mean: ", print_test_metric_mean)
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Dataset
    parser.add_argument('--Source_Dataset', type=str, default='BKAI',
                        help='BKAI/CVC-ClinicDB/ETIS-LaribPolypDB/Kvasir-SEG')
    parser.add_argument('--Target_Dataset', type=list)

    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--image_size', type=int, default=352)

    # Model
    parser.add_argument('--in_ch', type=int, default=3)
    parser.add_argument('--out_ch', type=int, default=1)

    # Optimizer
    parser.add_argument('--optimizer', type=str, default='Adam', help='SGD/Adam/AdamW')
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--momentum', type=float, default=0.99)  # momentum in SGD
    parser.add_argument('--beta1', type=float, default=0.9)  # beta1 in Adam
    parser.add_argument('--beta2', type=float, default=0.99)  # beta2 in Adam
    parser.add_argument('--weight_decay', type=float, default=0.00)

    # Training
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--iters', type=int, default=1)

    # Hyperparameters in prompt
    parser.add_argument('--prompt_alpha', type=float, default=0.01)

    # Path
    parser.add_argument('--path_save_log', type=str, default='./logs/')
    parser.add_argument('--model_root', type=str, default='./models/')
    parser.add_argument('--dataset_root', type=str, default='/media/userdisk0/zychen/Datasets/Polyp')

    # Cuda (default: the first available device)
    parser.add_argument('--device', type=str, default='cuda:0')

    config = parser.parse_args()

    config.Target_Dataset = ['BKAI', 'CVC-ClinicDB', 'ETIS-LaribPolypDB', 'Kvasir-SEG']

    config.Target_Dataset.remove(config.Source_Dataset)

    TTA = MGIPT(config)
    TTA.run()