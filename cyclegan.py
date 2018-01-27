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

from skimage.io import imsave
from skimage.transform import rescale

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
              vis_scale_factor=1.,
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
        fake_A_pool = util.ImagePool(num_pool)
        fake_B_pool = util.ImagePool(num_pool)
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
                key:[] for key in stats_keys if key not in ['epoch', 'time', 'lr_g', 'lr_da', 'lr_db']
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
                n_iters = min(len(itr_A),len(itr_B))
                if verbose:
                    pbar = tqdm(total=n_iters)
                # we will use rnd_b to find a random img to dump
                rnd_b = np.random.randint(0, n_iters)
                #rnd_b = 1
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
                    ###################
                    # TRAIN GENERATOR #
                    ###################
                    ones_da = torch.ones(d_a_fake.size())
                    ones_db = torch.ones(d_b_fake.size())
                    if self.gpu_mode:
                        ones_da, ones_db = ones_da.cuda(), ones_db.cuda()
                    ones_da, ones_db = Variable(ones_da), Variable(ones_db)
                    btoa_gen_loss = self.mse_loss(d_a_fake, ones_da)
                    atob_gen_loss = self.mse_loss(d_b_fake, ones_db)
                    cycle_aba = torch.mean(torch.abs(A_real - atob_btoa))
                    cycle_bab = torch.mean(torch.abs(B_real - btoa_atob))
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
                    A_fake = fake_A_pool.query(btoa)
                    B_fake = fake_B_pool.query(atob)
                    d_a_fake = self.d_a(A_fake)
                    d_b_fake = self.d_b(B_fake)
                    ones_da_real, zeros_da_fake = torch.ones(d_a_real.size()), torch.zeros(d_a_fake.size())
                    ones_db_real, zeros_db_fake = torch.ones(d_b_real.size()), torch.zeros(d_b_fake.size())
                    if self.gpu_mode:
                        ones_da_real, zeros_da_fake, ones_db_real, zeros_db_fake = \
                                    ones_da_real.cuda(), zeros_da_fake.cuda(), ones_db_real.cuda(), zeros_db_fake.cuda()
                    ones_da_real, zeros_da_fake, ones_db_real, zeros_db_fake = \
                                    Variable(ones_da_real), Variable(zeros_da_fake), Variable(ones_db_real), Variable(zeros_db_fake)
                    # disc A loss = squared_error(d_a_real, 1).mean() + squared_error(d_a_fake, 0).mean()
                    d_a_loss = (self.mse_loss(d_a_real, ones_da_real) + self.mse_loss(d_a_fake, zeros_da_fake)) * 0.5
                    # disc B loss = squared_error(d_b_real, 1).mean() + squared_error(d_b_fake, 0).mean()
                    d_b_loss = (self.mse_loss(d_b_real, ones_db_real) + self.mse_loss(d_b_fake, zeros_db_fake)) * 0.5
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
                    #########
                    if b == rnd_b:
                        # print out example from val set to disk
                        outs = [A_real, atob, atob_btoa, B_real, btoa, btoa_atob]
                        outs_np = [ x.data.cpu().numpy() for x in outs ]
                        shp = outs_np[0].shape[-1]
                        # possible that A_real.bs != B_real.bs
                        bs = np.min([outs_np[0].shape[0], outs_np[3].shape[0]])
                        grid = np.zeros((shp*bs, shp*6, 3))
                        for j in range(bs):
                            for i in range(6):
                                grid[j*shp:(j+1)*shp,i*shp:(i+1)*shp,:] = util.convert_to_rgb(outs_np[i][j])
                        imsave(arr=rescale(grid, scale=vis_scale_factor), fname="%s/%i_%s.png" % (result_dir, epoch+1, mode))
                if verbose:
                    pbar.close()
                #########
                for key in tmp_stats:
                    stats[key] = np.mean( tmp_stats[key] )
            stats['lr_g'] = self.optim_g.state_dict()['param_groups'][0]['lr']
            stats['lr_da'] = self.optim_d_a.state_dict()['param_groups'][0]['lr']
            stats['lr_db'] = self.optim_d_b.state_dict()['param_groups'][0]['lr']            
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

if __name__ == '__main__':
    from architectures.image2image import Generator, Discriminator
    
    net = CycleGAN(
        gen_fn=Generator,
        disc_fn=Discriminator,
        gen_fn_params={'input_dim':3, 'num_filter':32, 'output_dim':3, 'num_resnet':6},
        disc_fn_params={'input_dim':3, 'num_filter':64, 'output_dim':1},
        gpu_mode='detect',
        opt_d_args={'lr':0.0002, 'betas':(0.5, 0.999)},
        opt_g_args={'lr':0.0002, 'betas':(0.5, 0.999)},
    )

    # not sure if this is necessary??
    net.g_atob.normal_weight_init(mean=0.0, std=0.02)
    net.g_btoa.normal_weight_init(mean=0.0, std=0.02)
    net.d_a.normal_weight_init(mean=0.0, std=0.02)
    net.d_b.normal_weight_init(mean=0.0, std=0.02)
    
    bs = 1
    transforms = transforms.Compose([
        transforms.ToTensor(), transforms.Normalize(mean=(0.5,0.5,0.5), std=(0.5,0.5,0.5))
    ])
    ds_train_a = util.DatasetFromFolder("/data/lisa/data/beckhamc/cyclegan/datasets/horse2zebra/trainA/", resize_scale=286,
                                        crop_size=256, fliplr=True, transform=transforms)
    it_train_a = DataLoader(ds_train_a, batch_size=bs, shuffle=True)
    ds_train_b = util.DatasetFromFolder("/data/lisa/data/beckhamc/cyclegan/datasets/horse2zebra/trainB/", resize_scale=286,
                                        crop_size=256, fliplr=True, transform=transforms)
    it_train_b = DataLoader(ds_train_b, batch_size=bs, shuffle=True)
    
    ds_valid_a = util.DatasetFromFolder("/data/lisa/data/beckhamc/cyclegan/datasets/horse2zebra/testA/", resize_scale=286,
                                        crop_size=256, fliplr=True, transform=transforms)
    it_valid_a = DataLoader(ds_valid_a, batch_size=bs, shuffle=True)
    ds_valid_b = util.DatasetFromFolder("/data/lisa/data/beckhamc/cyclegan/datasets/horse2zebra/testB/", resize_scale=286,
                                        crop_size=256, fliplr=True, transform=transforms)
    it_valid_b = DataLoader(ds_valid_b, batch_size=bs, shuffle=True)


    #name = "cg_horse2zebra_gnf64_b9"
    name = "cg_horse2zebra_fixed"
    print net
    #net.load("models/%s/40.pkl" % name)
    net.train(
        itr_a_train=it_train_a,
        itr_b_train=it_train_b,
        itr_a_valid=it_valid_a,
        itr_b_valid=it_valid_b,
        epochs=100,
        model_dir="models/%s" % name,
        result_dir="results/%s" % name,
    )
    
