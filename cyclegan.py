import torch, time, os, pickle
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import ImageFolder
from torchvision import datasets, transforms
from collections import OrderedDict
import os
import sys
import math
from tqdm import tqdm
import h5py
import util
import itertools

class ImagePool():
    """
    Courtesy of:
    https://github.com/togheppi/CycleGAN/blob/master/utils.py
    """
    def __init__(self, pool_size):
        self.pool_size = pool_size
        if self.pool_size > 0:
            self.num_imgs = 0
            self.images = []

    def query(self, images):
        if self.pool_size == 0:
            return images
        return_images = []
        for image in images.data:
            image = torch.unsqueeze(image, 0)
            if self.num_imgs < self.pool_size:
                self.num_imgs = self.num_imgs + 1
                self.images.append(image)
                return_images.append(image)
            else:
                p = np.random.uniform(0, 1)
                if p > 0.5:
                    random_id = np.random.randint(0, self.pool_size-1)
                    tmp = self.images[random_id].clone()
                    self.images[random_id] = image
                    return_images.append(tmp)
                else:
                    return_images.append(image)
        return_images = Variable(torch.cat(return_images, 0))
        return return_images

class CycleGAN():
    def num_parameters(self, net):
        return np.sum([ np.prod(np.asarray(elem.size())) for elem in net.parameters() ])
    def __str__(self):
        g_summary = str(self.g_atob) + \
                    "\n# parameters for each G:" + str(self.num_parameters(self.g_atob))
        d_summary = str(self.d_a) + \
                    "\n# parameters for each D:" + str(self.num_parameters(self.d_a))
        return g_summary + "\n" + d_summary
    def __init__(self,
                 gen_fn,
                 disc_fn,
                 gen_fn_params={},
                 disc_fn_params={},
                 opt_g=optim.Adam, opt_g_args={},
                 opt_d=optim.Adam, opt_d_args={},
                 lamb=10.,
                 hooks={},
                 gpu_mode='detect'):
        assert gpu_mode in [True, False, 'detect']
        if gpu_mode == 'detect':
            gpu_mode = True if torch.cuda.is_available() else False
        self.lamb = lamb
        #######################
        # NETWORK DEFINITIONS #
        #######################
        self.g_atob = gen_fn(**gen_fn_params)
        self.g_btoa = gen_fn(**gen_fn_params)
        self.d_a = disc_fn(**disc_fn_params)
        self.d_b = disc_fn(**disc_fn_params)
        ##########
        # LOSSES #
        ##########
        self.mse_loss = torch.nn.MSELoss()
        #self.l1_loss = torch.nn.L1Loss()
        ##############
        # OPTIMISERS #
        ##############
        self.optim_g = opt_g( itertools.chain(self.g_atob.parameters(), self.g_btoa.parameters()), **opt_g_args)
        self.optim_d_a = opt_d( self.d_a.parameters(), **opt_d_args)
        self.optim_d_b = opt_d( self.d_b.parameters(), **opt_d_args)
        #########
        # OTHER #
        #########
        self.hooks = hooks
        self.gpu_mode = gpu_mode
        if self.gpu_mode:
            self.g_atob.cuda()
            self.g_btoa.cuda()
            self.d_a.cuda()
            self.d_b.cuda()
            self.mse_loss.cuda()
    def print_gpu_stats(self):
        try:
            from pynvml import nvmlInit, nvmlDeviceGetHandleByIndex, nvmlDeviceGetMemoryInfo
            nvmlInit()
            handle = nvmlDeviceGetHandleByIndex(0)
            totalMemory = nvmlDeviceGetMemoryInfo(handle)
            print "GPU memory used:", totalMemory.used / 1024. / 1024.
        except Exception as e:
            print "Was unable to detect amount of GPU memory used"
            print e
    def train(self,
              itr_a_train, itr_b_train,
              itr_a_valid, itr_b_valid,
              epochs, model_dir, result_dir,
              resume=False,
              save_every=1,
              scheduler=None,
              scheduler_args={},
              scheduler_metric='valid_loss',
              max_iters=-1,
              verbose=True):
        """
        """
        for folder_name in [model_dir, result_dir]:
            if folder_name is not None and not os.path.exists(folder_name):
                os.makedirs(folder_name)
        #if scheduler != None:
        #    scheduler = scheduler(self.optim, **scheduler_args)
        #else:
        #    scheduler = None
        f = open("%s/results.txt" % result_dir, "wb" if not resume else "a") if result_dir != None else None
        start_time = time.time()
        stats_keys = ['epoch',
                      'train_g_atob_loss', 'train_g_btoa_loss', 'train_aba_loss', 'train_bab_loss',
                      'train_da_loss', 'train_db_loss',
                      'valid_g_atob_loss', 'valid_g_btoa_loss', 'valid_aba_loss', 'valid_bab_loss',
                      'valid_da_loss', 'valid_db_loss',
                      'lr_g', 'lr_da', 'lr_db', 'time']
        num_pool = 50
        fake_A_pool = ImagePool(num_pool)
        fake_B_pool = ImagePool(num_pool)
        for epoch in range(epochs):
            stats = OrderedDict({})
            for key in stats_keys:
                stats[key] = None
            stats['epoch'] = epoch+1
            if (epoch+1) == 1:
                # if this is the start of training, print out / save the header
                if f != None and not resume:
                    f.write(",".join(stats.keys()) + "\n")
                if verbose:
                    print ",".join(stats.keys())                
            epoch_start_time = time.time()
            # accumulator to be able to compute averages
            tmp_stats = {
                key:[] for key in stats_keys if key not in ['epoch', 'time']
            }
            for idx, (itr_A, itr_B) in enumerate([ (itr_a_train, itr_b_train), (itr_a_valid, itr_b_valid) ]):
                mode = 'train' if idx==0 else 'valid'
                if mode == 'train':
                    self.g_atob.train()
                    self.g_btoa.train()
                    self.d_a.train()
                    self.d_b.train()
                else:
                    self.g_atob.eval()
                    self.g_btoa.eval()
                    self.d_a.eval()
                    self.d_b.eval()
                if verbose:
                    pbar = tqdm( total=min(len(itr_A),len(itr_B)) )
                for b, (A_real, B_real) in enumerate(zip(itr_A, itr_B)):
                    if max_iters > 0 and b == max_iters:
                        break
                    if verbose:
                        pbar.update(1)
                    # prep data
                    A_real, B_real = A_real.float(), B_real.float()
                    if self.gpu_mode:
                        A_real, B_real = A_real.cuda(), B_real.cuda()
                    A_real, B_real = Variable(A_real), Variable(B_real)
                    # compute outputs
                    atob = self.g_atob(A_real)
                    btoa = self.g_btoa(B_real)
                    atob_btoa = self.g_btoa(atob)
                    btoa_atob = self.g_atob(btoa)
                    d_a_fake = self.d_a(btoa)
                    d_b_fake = self.d_b(atob)
                    # constants
                    ones, zeros = torch.ones(d_a_fake.size()), torch.zeros(d_a_fake.size())
                    if self.gpu_mode:
                        ones, zeros = ones.cuda(), zeros.cuda()
                    ones, zeros = Variable(ones), Variable(zeros)
                    ###################
                    # TRAIN GENERATOR #
                    ###################
                    # gen A loss = squared_error(d_a_fake, 1).mean()
                    btoa_gen_loss = self.mse_loss(d_a_fake, ones)
                    # gen B loss = squared_error(d_b_fake, 1).mean()
                    atob_gen_loss = self.mse_loss(d_b_fake, ones)
                    # atob cycle
                    cycle_aba = torch.mean(torch.abs(A_real - atob_btoa))
                    # btoa cycle
                    cycle_bab = torch.mean(torch.abs(B_real - btoa_atob))
                    # backprop for G
                    g_tot_loss = atob_gen_loss + btoa_gen_loss + self.lamb*cycle_aba + self.lamb*cycle_bab
                    if mode == 'train':
                        self.optim_g.zero_grad()
                        g_tot_loss.backward()
                        self.optim_g.step()
                    #######################
                    # TRAIN DISCRIMINATOR #
                    #######################
                    d_a_real = self.d_a(A_real)
                    d_b_real = self.d_b(B_real)
                    d_a_fake = fake_A_pool.query(d_a_fake)
                    d_b_fake = fake_B_pool.query(d_b_fake)
                    # disc A loss = squared_error(d_a_real, 1).mean() + squared_error(d_a_fake, 0).mean()
                    d_a_loss = self.mse_loss(d_a_real, ones) + self.mse_loss(d_a_fake, zeros) 
                    # disc B loss = squared_error(d_b_real, 1).mean() + squared_error(d_b_fake, 0).mean()
                    d_b_loss = self.mse_loss(d_b_real, ones) + self.mse_loss(d_b_fake, zeros)
                    if mode == 'train':
                        # backprop for D_A
                        self.optim_d_a.zero_grad()
                        d_a_loss.backward()
                        self.optim_d_a.step()
                        # backprop for D_B
                        self.optim_d_b.zero_grad()
                        d_b_loss.backward()
                        self.optim_d_b.step()
                    if b == 0 and epoch == 0 and self.gpu_mode:
                        self.print_gpu_stats()
                    #########
                    tmp_stats['%s_g_atob_loss' % mode].append(atob_gen_loss.data[0])
                    tmp_stats['%s_g_btoa_loss' % mode].append(btoa_gen_loss.data[0])
                    tmp_stats['%s_aba_loss' % mode].append(cycle_aba.data[0])
                    tmp_stats['%s_bab_loss' % mode].append(cycle_bab.data[0])
                    tmp_stats['%s_da_loss' % mode].append(d_a_loss.data[0])
                    tmp_stats['%s_db_loss' % mode].append(d_b_loss.data[0])
                if verbose:
                    pbar.close()
                #########
                for key in tmp_stats:
                    stats[key] = np.mean( tmp_stats[key] )
            stats['lr_g'] = self.optim_g.state_dict()['param_groups'][0]['lr']
            stats['lr_a'] = self.optim_d_a.state_dict()['param_groups'][0]['lr']
            stats['lr_b'] = self.optim_d_b.state_dict()['param_groups'][0]['lr']            
            stats['time'] = time.time() - epoch_start_time
            # update the learning rate scheduler, if applicable
            #if scheduler != None:
            #    scheduler.step(stats[scheduler_metric])
            str_ = ",".join([ str(stats[key]) for key in stats ])
            print str_
            if f != None:
                f.write(str_ + "\n")
                f.flush()
            if (epoch+1) % save_every == 0 and model_dir != None:
                self.save( filename="%s/%i.pkl" % (model_dir, epoch+1) )
        if f != None:
            f.close()       
    def save(self, filename):
        torch.save(
            (self.g_atob.state_dict(), self.g_btoa.state_dict(), self.d_a.state_dict(), self.d_b.state_dict()),
            filename)
    def load(self, filename, cpu=False):
        if cpu:
            map_location = lambda storage, loc: storage
        else:
            map_location = None
        g_atob, g_btoa, d_a, d_b = torch.load(filename, map_location=map_location)
        self.g_atob.load_state_dict(g_atob)
        self.g_btoa.load_state_dict(g_btoa)
        self.d_a.load_state_dict(d_a)
        self.d_b.load_state_dict(d_b)


class ImageFolderWithoutClass(ImageFolder):
    def __getitem__(self, index):
        img, _ = super(ImageFolderWithoutClass, self).__getitem__(index)
        return img


if __name__ == '__main__':
    from architectures.image2image import Generator, Discriminator
    
    net = CycleGAN(
        gen_fn=Generator,
        disc_fn=Discriminator,
        gen_fn_params={'input_dim':3, 'num_filter':32, 'output_dim':3, 'num_resnet':6},
        disc_fn_params={'input_dim':3, 'num_filter':64, 'output_dim':3},
        gpu_mode='detect',
        opt_d_args={'lr':0.0002, 'betas':(0.5, 0.999)},
        opt_g_args={'lr':0.0002, 'betas':(0.5, 0.999)},
    )
    bs = 1
    ds_train_a = util.DatasetFromFolder("/data/lisa/data/beckhamc/cyclegan/datasets/horse2zebra/trainA/", resize_scale=286,
                                        crop_size=256, fliplr=True, transform=transforms.ToTensor())
    it_train_a = DataLoader(ds_train_a, batch_size=bs, shuffle=True, num_workers=0)
    ds_train_b = util.DatasetFromFolder("/data/lisa/data/beckhamc/cyclegan/datasets/horse2zebra/trainB/", resize_scale=286,
                                        crop_size=256, fliplr=True, transform=transforms.ToTensor())
    it_train_b = DataLoader(ds_train_b, batch_size=bs, shuffle=True, num_workers=0)
    net.train(
        itr_a_train=it_train_a,
        itr_b_train=it_train_b,
        itr_a_valid=it_train_a,
        itr_b_valid=it_train_b,
        epochs=10,
        model_dir="models/cg_horse2zebra",
        result_dir="results/cg_horse2zebra"
    )
    
