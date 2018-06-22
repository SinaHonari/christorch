import numpy as np
import torch
from torch import optim
from torch import nn
from .base import GAN
from itertools import chain

class ALI(GAN):

    def __init__(self,
                 gx, gz, dx, dxz,
                 z_dim,
                 opt_g=optim.Adam, opt_d=optim.Adam,
                 opt_d_args={'lr': 0.0002, 'betas': (0.5, 0.999)},
                 opt_g_args={'lr': 0.0002, 'betas': (0.5, 0.999)},
                 lamb=0.,
                 dnorm=None,
                 handlers=[],
                 scheduler_fn=None,
                 scheduler_args={},
                 use_cuda='detect'):
        assert use_cuda in [True, False, 'detect']
        if use_cuda == 'detect':
            use_cuda = True if torch.cuda.is_available() else False
        self.z_dim = z_dim
        self.dnorm = dnorm
        optim_g = optim.Adam(chain(gx.parameters(),gz.parameters()),
                             **opt_g_args)
        optim_d = optim.Adam(chain(dx.parameters(),dxz.parameters()),
                             **opt_d_args)
        self.gx = gx
        self.gz = gz
        self.dx = dx
        self.dxz = dxz
        self.lamb = lamb
        self.optim = {
            'g': optim_g,
            'd': optim_d,
        }
        self.scheduler = {}
        if scheduler_fn is not None:
            for key in self.optim:
                self.scheduler[key] = scheduler_fn(
                    self.optim[key], **scheduler_args)
        self.handlers = handlers
        self.use_cuda = use_cuda
        if self.use_cuda:
            self.gx.cuda()
            self.gz.cuda()
            self.dx.cuda()
            self.dxz.cuda()
        self.last_epoch = 0

    def _train(self):
        self.gx.train()
        self.gz.train()
        self.dx.train()
        self.dxz.train()

    def _eval(self):
        self.gx.eval()
        self.gz.eval()
        self.dx.eval()
        self.dxz.eval()

    def sample(self, bs, seed=None):
        """Return a sample G(z)"""
        self._eval()
        with torch.no_grad():
            z_batch = self.sample_z(bs, seed=seed)
            gx = self.gx(z_batch)
        return gx

    def sample_z(self, bs, seed=None):
        """Return a sample z ~ p(z)"""
        if seed is not None:
            rnd_state = np.random.RandomState(seed)
            z = torch.from_numpy(
                rnd_state.normal(0, 1, size=(bs, self.z_dim, 1, 1))
            ).float()
        else:
            z = torch.from_numpy(
                np.random.normal(0, 1, size=(bs, self.z_dim, 1, 1))
            ).float()
        if self.use_cuda:
            z = z.cuda()
        return z
    
    def prepare_batch(self, x):
        return x
        
    def train_on_instance(self, z, x, **kwargs):
        batch_size = x.size(0)
        if self.use_cuda:
            x = x.cuda()
        if self.use_cuda:
            z = z.cuda()
        # forward
        x_fake = self.gx(z) # z -> x'
        encoded = self.gz(x) # x -> z'
        # reparameterisation trick
        eps = self.sample_z(batch_size)
        eps.require_grad = False
        z_enc = encoded[:, :self.z_dim] + \
                encoded[:, self.z_dim:].exp() * eps
        if self.lamb > 0:
            x_recon = self.gx(z_enc)
            recon = torch.mean((x_recon - x)**2)
        dx_true = self.dx(x)
        dx_fake = self.dx(x_fake)
        d_true = self.dxz(torch.cat((dx_true, z_enc), dim=1))
        d_fake = self.dxz(torch.cat((dx_fake, z), dim=1))
        # compute loss
        softplus = nn.Softplus()
        loss_d = torch.mean(softplus(-d_true) + softplus(d_fake))
        loss_g = torch.mean(softplus(d_true) + softplus(-d_fake))
        if self.lamb > 0:
            loss_g += self.lamb*recon
        # backward & update params
        self.dx.zero_grad()
        self.dxz.zero_grad()
        loss_d.backward(retain_graph=True)
        self.optim['d'].step()
        self.gx.zero_grad()
        self.gz.zero_grad()
        loss_g.backward()
        self.optim['g'].step()
        losses = {
            'loss_d': loss_d.item(),
            'loss_g': loss_g.item()
        }
        if self.lamb > 0:
            losses['recon'] = recon.item()
        return losses, {}

    def eval_on_instance(self, z, x, **kawrgs):
        pass

    def save(self, filename, epoch):
        dd = {}
        dd['gx'] = self.gx.state_dict()
        dd['gz'] = self.gz.state_dict()        
        dd['dx'] = self.dx.state_dict()
        dd['dxz'] = self.dxz.state_dict()        
        for key in self.optim:
            dd['optim_' + key] = self.optim[key].state_dict()
        dd['epoch'] = epoch
        torch.save(dd, filename)

    def load(self, filename):
        """
        ignore_d: if `True`, then don't load in the
          discriminator.
        """
        if not self.use_cuda:
            map_location = lambda storage, loc: storage
        else:
            map_location = None
        dd = torch.load(filename,
                        map_location=map_location)
        self.gx.load_state_dict(dd['gx'])
        self.gz.load_state_dict(dd['gz'])
        self.dx.load_state_dict(dd['dx'])
        self.dxz.load_state_dict(dd['dxz'])

        for key in self.optim:
            self.optim[key].load_state_dict(dd['optim_'+key])
        self.last_epoch = dd['epoch']
