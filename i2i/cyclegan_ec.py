from __future__ import print_function
import numpy as np
import torch
from collections import OrderedDict
from torch.autograd import Variable, grad
from .. import util

from .cyclegan import CycleGAN

from skimage.io import imsave
from skimage.transform import rescale, resize

def dump_visualisation_eg(out_folder, save_every_iter=100, scale_factor=1.):
    """
    """
    def _fn(losses, inputs, outputs, kwargs):
        iter_n = kwargs['iter'] - 1
        if save_every_iter is not None:
            if iter_n % save_every_iter != 0:
                return
        else:
            if iter_n != 0:
                return
        A_real = inputs[0].data.cpu().numpy()
        B_real = inputs[1].data.cpu().numpy()
        atob, atob_btoa, atob_zero_btoa, btoa, btoa_atob = \
            [elem.data.cpu().numpy() for elem in outputs.values()]
        n_ch = atob.shape[1]-1
        # A
        # A->B (actual)
        # A->B hidden channel
        # A->B->A (cycle)
        # A->B->zero->A (remove encoding)
        # B
        # B->A
        # B->A->B
        # B->A->B hidden channel
        outs_np = [A_real,
                   atob[:, 0:n_ch],
                   atob[:, n_ch:(n_ch+1)],
                   atob_btoa,
                   atob_zero_btoa,
                   B_real[:, 0:n_ch],
                   btoa,
                   btoa_atob[:, 0:n_ch],
                   btoa_atob[:, n_ch:(n_ch+1)]]
        w, h = outs_np[0].shape[-1], outs_np[0].shape[-2]
        bs = np.min([outs_np[0].shape[0], outs_np[3].shape[0]])
        grid = np.zeros((h*bs, w*len(outs_np), 3))
        for j in range(bs):
            for i in range(len(outs_np)):
                # If n_channels >= 3, then get the first three channels.
                # If n_channels < 3, then get the first channel.
                this_img = outs_np[i][j]
                img_to_write = util.convert_to_rgb(this_img,
                                                   is_grayscale=False)
                grid[j*h:(j+1)*h, i*w:(i+1)*w, :] = img_to_write
        imsave(arr=rescale(grid, scale=scale_factor),
               fname="%s/%s_%i_%i.png" % (out_folder,
                                          kwargs['mode'],
                                          kwargs['epoch'],
                                          kwargs['iter']))
    return _fn


class CycleGAN_EC(CycleGAN):

    def __init__(self, *args, **kwargs):
        if 'lamb_key' in kwargs:
            self.lamb_key = kwargs['lamb_key']
        else:
            self.lamb_key = 0.01
        kwargs.pop('lamb_key', None)
        super(CycleGAN_EC, self).__init__(*args, **kwargs)

    def prepare_batch(self, A_real, B_real):
        A_real, B_real = A_real.float(), B_real.float()
        # We need to add an extra 'dummy' channel to B_real
        shp = list(B_real.shape)
        shp[1] = 1
        B_real = torch.cat((B_real, torch.zeros(shp)), dim=1)
        if self.use_cuda:
            A_real, B_real = A_real.cuda(), B_real.cuda()
        A_real, B_real = Variable(A_real), Variable(B_real)
        return A_real, B_real

    def compute_g_losses_aba(self, A_real, atob, atob_btoa):
        """Return all the losses related to generation"""
        n_ch = atob.shape[1] - 1
        atob_gen_loss = self.mse(self.d_b(atob[:, 0:n_ch]), 1)
        cycle_aba = torch.mean(torch.abs(A_real - atob_btoa))
        return atob_gen_loss, cycle_aba

    def compute_g_losses_bab(self, B_real, btoa, btoa_atob):
        """Return all the losses related to generation"""
        btoa_gen_loss = self.mse(self.d_a(btoa), 1)
        cycle_bab = torch.mean(torch.abs(B_real - btoa_atob))
        return btoa_gen_loss, cycle_bab

    def compute_d_losses(self, A_real, atob, B_real, btoa):
        """Return all losses related to discriminator"""
        fake_a = self.fake_A_pool.query(btoa)
        fake_b = self.fake_B_pool.query(atob)
        n_ch = fake_b.shape[1] - 1
        d_a_fake = self.d_a(fake_a)
        d_b_fake = self.d_b(fake_b[:, 0:n_ch])
        d_a_loss = 0.5*(self.mse(self.d_a(A_real), 1) +
                        self.mse(d_a_fake, 0))
        d_b_loss = 0.5*(self.mse(self.d_b(B_real[:, 0:n_ch]), 1) +
                        self.mse(d_b_fake, 0))
        return d_a_loss, d_b_loss

    def get_noise(self, x, std):
        noise = torch.randn(x.size()).normal_(0., std)
        n_ch = x.size()[1]-1
        # Don't add noise to the cycle key.
        noise[:,n_ch,:,:] = 0.
        noise = Variable(noise, requires_grad=False)
        if self.use_cuda:
            noise = noise.cuda()
        return noise
    
    def train_on_instance(self, A_real, B_real):
        """Train the network on a single example"""
        self._train()
        # a(k) -> b(k+1) -> a(k)
        atob = self.g_atob(A_real)
        #atob_ck = (atob[:,-1,:,:]**2).mean()
        atob_ck = torch.mean(torch.abs(atob[:,-1]))
        if self.noise_std > 0.:
            atob_btoa = self.g_btoa(atob + self.get_noise(atob,
                                                          self.noise_std))
        else:
            atob_btoa = self.g_btoa(atob)
        atob_gen_loss, cycle_aba = self.compute_g_losses_aba(
            A_real, atob, atob_btoa)
        g_tot_loss = atob_gen_loss + self.lamb*cycle_aba + self.lamb_key*atob_ck
        '''
        if self.d_eps:
            mask = torch.ones(atob.size())
            mask[:, -1, :, :] = 0.
            if self.use_cuda:
                mask = mask.cuda()
            mask = Variable(mask, requires_grad=False)
            atob_zeroed = atob*mask
            atob_btoa_zeroed = self.g_btoa(atob_zeroed)
            extra_loss = self.mse(self.d_a(atob_btoa_zeroed), 1)
            g_tot_loss += extra_loss
        '''
        self.optim['g'].zero_grad()
        g_tot_loss.backward()
        # b(k+1) -> a(k) -> b(k+1)
        btoa = self.g_btoa(B_real)
        if self.noise_std > 0.:
            btoa_atob = self.g_atob(btoa + self.get_noise(btoa,
                                                          self.noise_std))
        else:
            btoa_atob = self.g_atob(btoa)
        eps_loss = (btoa_atob[:, -1, :, :]**2).mean()
        btoa_gen_loss, cycle_bab = self.compute_g_losses_bab(
            B_real, btoa, btoa_atob)
        g_tot_loss = btoa_gen_loss + self.lamb*cycle_bab + eps_loss
        g_tot_loss.backward()
        d_a_loss, d_b_loss = self.compute_d_losses(A_real, atob, B_real, btoa)
        '''
        if self.d_eps:
            d_a_loss += self.mse(self.d_a(atob_btoa_zeroed.detach()), 0)
        '''
        self.optim['d_a'].zero_grad()
        self.optim['d_b'].zero_grad()
        d_a_loss.backward()
        d_b_loss.backward()
        if self.dnorm is not None and self.dnorm > 0.:
            gp_a, gp_b = self.compute_d_norms(atob, btoa)
            (gp_a*self.dnorm).backward(retain_graph=True)
            (gp_b*self.dnorm).backward(retain_graph=True)
        self.optim['g'].step()
        self.optim['d_a'].step()
        self.optim['d_b'].step()
        with torch.no_grad():
            # Zero out the hidden encoded information in the
            # last channel, then decode.
            atob_zero = atob.clone()
            atob_zero[:, -1, :, :] *= 0.
            atob_zero_btoa = self.g_btoa(atob_zero)
        losses = {
            'atob_gen': atob_gen_loss.item(),
            'cycle_aba': cycle_aba.item(),
            'btoa_gen': btoa_gen_loss.item(),
            'cycle_bab': cycle_bab.item(),
            'atob_ck': atob_ck.item(),
            #'extra_loss': extra_loss.item(),
            'eps': eps_loss.item(),
            'd_a': d_a_loss.item(),
            'd_b': d_b_loss.item()
        }
        if self.dnorm is not None and self.dnorm > 0.:
            losses['gp_a'] = gp_a.item()
            losses['gp_b'] = gp_b.item()
        outputs = {
            'atob': atob.detach(),
            'atob_btoa': atob_btoa.detach(),
            'atob_zero_btoa': atob_zero_btoa,
            'btoa': btoa.detach(),
            'btoa_atob': btoa_atob.detach()
        }
        return losses, outputs

    def _get_stats(self, dict_, mode):
        """
        From a dict of training/valid statistics, create a
          summarised dict for use with the progress bar.
        """
        allowed_keys = ['atob_gen', 'btoa_gen', 'd_a', 'd_b',
                        'gp_a', 'gp_b', 'eps', 'atob_ck']
        allowed_keys = ['%s_%s' % (mode, key) for key in allowed_keys]
        stats = OrderedDict({})
        for key in dict_.keys():
            if key in allowed_keys:
                stats[key] = np.mean(dict_[key])
        return stats


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
