from __future__ import print_function

import time, os, pickle, sys, math
import numpy as np
# torch imports
import torch
import torch.optim as optim
from torch.autograd import Variable
# torchvision
from collections import OrderedDict

from tqdm import tqdm
from . import util
import itertools

import logging

class CycleGAN():
    def num_parameters(self, net):
        return util.count_params(net)

    def __str__(self):
        g_summary = str(self.g_atob) + \
                    "\n# parameters for each G:" + str(self.num_parameters(self.g_atob))
        d_summary = str(self.d_a) + \
                    "\n# parameters for each D:" + str(self.num_parameters(self.d_a))
        return g_summary + "\n" + d_summary

    def __init__(self,
                 gen_atob_fn,
                 disc_a_fn,
                 gen_btoa_fn,
                 disc_b_fn,
                 opt_g=optim.Adam, opt_g_args={},
                 opt_d=optim.Adam, opt_d_args={},
                 lamb=10.,
                 beta=5.,
                 pool_size=50,
                 handlers=[],
                 use_cuda='detect'):
        assert use_cuda in [True, False, 'detect']
        if use_cuda == 'detect':
            use_cuda = True if torch.cuda.is_available() else False
        self.lamb = lamb
        self.beta = beta
        self.g_atob = gen_atob_fn
        self.g_btoa = gen_btoa_fn
        self.d_a = disc_a_fn
        self.d_b = disc_b_fn
        self.optim_g = opt_g(
            itertools.chain(
                self.g_atob.parameters(),
                self.g_btoa.parameters()),
            **opt_g_args)
        self.optim_d_a = opt_d( self.d_a.parameters(), **opt_d_args)
        self.optim_d_b = opt_d( self.d_b.parameters(), **opt_d_args)
        self.handlers = handlers
        self.use_cuda = use_cuda
        self.fake_A_pool = util.ImagePool(pool_size)
        self.fake_B_pool = util.ImagePool(pool_size)
        if self.use_cuda:
            self.g_atob.cuda()
            self.g_btoa.cuda()
            self.d_a.cuda()
            self.d_b.cuda()

    def mse(self, prediction, target):
        if not hasattr(target, '__len__'):
            target = torch.ones_like(prediction)*target
            if prediction.is_cuda:
                target = target.cuda()
            target = Variable(target)
        return torch.nn.MSELoss()(prediction, target)

    def compute_g_losses_aba(self, A_real, atob, atob_btoa):
        """Return all the losses related to generation"""
        atob_gen_loss = self.mse(self.d_b(atob), 1)
        cycle_aba = torch.mean(torch.abs(A_real - atob_btoa))
        cycle_id_a = torch.mean(torch.abs(A_real - self.g_btoa(A_real)))
        return atob_gen_loss, cycle_aba, cycle_id_a

    def compute_g_losses_bab(self, B_real, btoa, btoa_atob):
        """Return all the losses related to generation"""
        btoa_gen_loss = self.mse(self.d_a(btoa), 1)
        cycle_bab = torch.mean(torch.abs(B_real - btoa_atob))
        cycle_id_b = torch.mean(torch.abs(B_real - self.g_atob(B_real)))
        return btoa_gen_loss, cycle_bab, cycle_id_b

    def compute_d_losses(self, A_real, atob, B_real, btoa):
        """Return all losses related to discriminator"""
        d_a_loss = 0.5*(self.mse(self.d_a(A_real), 1) +
                        self.mse(self.d_a(self.fake_A_pool.query(btoa)), 0))
        d_b_loss = 0.5*(self.mse(self.d_b(B_real), 1) +
                        self.mse(self.d_b(self.fake_B_pool.query(atob)), 0))
        return d_a_loss, d_b_loss

    def _zip(self, A, B):
        if sys.version[0] == '2':
            from itertools import izip
            return izip(A, B)
        else:
            return zip(A, B)

    def prepare_batch(self, A_real, B_real):
        A_real, B_real = A_real.float(), B_real.float()
        if self.use_cuda:
            A_real, B_real = A_real.cuda(), B_real.cuda()
        A_real, B_real = Variable(A_real), Variable(B_real)
        return A_real, B_real

    def _train(self):
        self.g_atob.train()
        self.g_btoa.train()
        self.d_a.train()
        self.d_b.train()

    def _eval(self):
        self.g_atob.eval()
        self.g_btoa.eval()
        self.d_a.eval()
        self.d_b.eval()

    def train_on_instance(self, A_real, B_real):
        """Train the network on a single example"""
        self._train()
        atob = self.g_atob(A_real)
        atob_btoa = self.g_btoa(atob)
        atob_gen_loss, cycle_aba, cycle_id_a = self.compute_g_losses_aba(
            A_real, atob, atob_btoa)
        g_tot_loss = atob_gen_loss + self.lamb*cycle_aba + self.beta*cycle_id_a
        self.optim_g.zero_grad()
        g_tot_loss.backward()
        self.optim_g.step()
        btoa = self.g_btoa(B_real)
        btoa_atob = self.g_atob(btoa)
        btoa_gen_loss, cycle_bab, cycle_id_b = self.compute_g_losses_bab(
            B_real, btoa, btoa_atob)
        g_tot_loss = btoa_gen_loss + self.lamb*cycle_bab + self.beta*cycle_id_b
        self.optim_g.zero_grad()
        g_tot_loss.backward()
        self.optim_g.step()
        d_a_loss, d_b_loss = self.compute_d_losses(A_real, atob, B_real, btoa)
        self.optim_d_a.zero_grad()
        d_a_loss.backward()
        self.optim_d_a.step()
        self.optim_d_b.zero_grad()
        d_b_loss.backward()
        self.optim_d_b.step()
        losses = {
            'atob_gen': atob_gen_loss.item(),
            'cycle_aba': cycle_aba.item(),
            'cycle_id_a': cycle_id_a.item(),
            'btoa_gen': btoa_gen_loss.item(),
            'cycle_bab': cycle_bab.item(),
            'cycle_id_b': cycle_id_b.item(),
            'd_a': d_a_loss.item(),
            'd_b': d_b_loss.item()
        }
        outputs = {
            'atob': atob.detach(),
            'atob_btoa': atob_btoa.detach(),
            'btoa': btoa.detach(),
            'btoa_atob': btoa_atob.detach()
        }
        return losses, outputs

    def eval_on_instance(self, A_real, B_real):
        """Train the network on a single example"""
        self._eval()
        with torch.no_grad():
            atob = self.g_atob(A_real)
            atob_btoa = self.g_btoa(atob)
            atob_gen_loss, cycle_aba, cycle_id_a = self.compute_g_losses_aba(
                A_real, atob, atob_btoa)
            btoa = self.g_btoa(B_real)
            btoa_atob = self.g_atob(btoa)
            btoa_gen_loss, cycle_bab, cycle_id_b = self.compute_g_losses_bab(
                B_real, btoa, btoa_atob)
            d_a_loss, d_b_loss = self.compute_d_losses(
                A_real, atob, B_real, btoa)
        losses = {
            'atob_gen': atob_gen_loss.item(),
            'cycle_aba': cycle_aba.item(),
            'cycle_id_a': cycle_id_a.item(),
            'btoa_gen': btoa_gen_loss.item(),
            'cycle_bab': cycle_bab.item(),
            'cycle_id_b': cycle_id_b.item(),
            'd_a': d_a_loss.item(),
            'd_b': d_b_loss.item()
        }
        outputs = {
            'atob': atob.detach(),
            'atob_btoa': atob_btoa.detach(),
            'btoa': btoa.detach(),
            'btoa_atob': btoa_atob.detach()
        }
        return losses, outputs

    def _get_postfix(self, dict_stats, mode):
        """Create a postfix string for progress bar"""
        allowed_keys = ['atob_gen', 'btoa_gen', 'd_a', 'd_b']
        allowed_keys = ['%s_%s' % (mode,key) for key in allowed_keys]
        stats = OrderedDict({})
        for key in dict_stats.keys():
            if key in allowed_keys:
                stats[key] = np.mean(dict_stats[key])
        return stats
    
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
        for folder_name in [model_dir, result_dir]:
            if folder_name is not None and not os.path.exists(folder_name):
                os.makedirs(folder_name)
        f_mode = 'w' if not resume else 'a'
        if result_dir is not None:
            f = open("%s/results.txt" % result_dir, f_mode)
        else:
            f = None
        for epoch in range(epochs):
            # Training
            epoch_start_time = time.time()
            if verbose:
                n_iters = min(len(itr_a_train), len(itr_b_train))
                pbar = tqdm(total=n_iters)
            train_dict = OrderedDict({'epoch': epoch})
            for b, (A_real, B_real) in enumerate(
                    self._zip(itr_a_train, itr_b_train)):
                A_real, B_real = self.prepare_batch(A_real, B_real)
                losses, outputs = self.train_on_instance(A_real, B_real)
                for key in losses:
                    this_key = 'train_%s' % key
                    if this_key not in train_dict:
                        train_dict[this_key] = []
                    train_dict[this_key].append(losses[key])
                pbar.update(1)
                pbar.set_postfix(self._get_postfix(train_dict, 'train'))
                for handler_fn in self.handlers:
                    handler_fn(losses, (A_real, B_real), outputs,
                               {'epoch':epoch, 'iter':b, 'mode':'train'})
            # Validation
            if verbose:
                pbar.close()
                n_iters = min(len(itr_a_valid), len(itr_b_valid))
                pbar = tqdm(total=n_iters)
            valid_dict = OrderedDict({})
            for b, (A_real, B_real) in enumerate(
                    self._zip(itr_a_valid, itr_b_valid)):
                A_real, B_real = self.prepare_batch(A_real, B_real)
                losses, outputs = self.eval_on_instance(A_real, B_real)
                for key in losses:
                    this_key = 'valid_%s' % key
                    if this_key not in valid_dict:
                        valid_dict[this_key] = []
                    valid_dict[this_key].append(losses[key])
                pbar.update(1)
                pbar.set_postfix(self._get_postfix(valid_dict, 'valid'))
                for handler_fn in self.handlers:
                    handler_fn(losses, (A_real, B_real), outputs,
                               {'epoch':epoch, 'iter':b, 'mode':'valid'})
            if verbose:
                pbar.close()
            all_dict = train_dict
            all_dict.update(valid_dict)
            for key in all_dict:
                all_dict[key] = np.mean(all_dict[key])
            all_dict['lr_g'] = \
                self.optim_g.state_dict()['param_groups'][0]['lr']
            all_dict['lr_da'] = \
                self.optim_d_a.state_dict()['param_groups'][0]['lr']
            all_dict['lr_db'] = \
                self.optim_d_b.state_dict()['param_groups'][0]['lr']
            all_dict['time'] = \
                time.time() - epoch_start_time
            str_ = ",".join([str(all_dict[key]) for key in all_dict])
            print(str_)
            if f is not None:
                if epoch == 0 and not resume:
                    # If we're not resuming, then write the header.
                    f.write(",".join(all_dict.keys()) + "\n")
                f.write(str_ + "\n")
                f.flush()
            if (epoch+1) % save_every == 0 and model_dir is not None:
                self.save(filename="%s/%i.pkl" % (model_dir, epoch+1))
        if f is not None:
            f.close()
    def save(self, filename):
        torch.save(
            (self.g_atob.state_dict(),
             self.g_btoa.state_dict(),
             self.d_a.state_dict(),
             self.d_b.state_dict()),
            filename)
    def load(self, filename):
        if not self.use_cuda:
            map_location = lambda storage, loc: storage
        else:
            map_location = None
        g_atob, g_btoa, d_a, d_b = torch.load(
            filename, map_location=map_location)
        self.g_atob.load_state_dict(g_atob)
        self.g_btoa.load_state_dict(g_btoa)
        self.d_a.load_state_dict(d_a)
        self.d_b.load_state_dict(d_b)
