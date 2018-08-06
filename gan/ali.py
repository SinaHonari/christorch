import numpy as np
import torch
from torch import optim
from torch import nn
from .base import GAN
from itertools import chain
from torch.autograd import grad

def bce(prediction, target):
    if not hasattr(target, '__len__'):
        target = torch.ones_like(prediction)*target
        if prediction.is_cuda:
            target = target.cuda()
    loss = torch.nn.BCELoss()
    if prediction.is_cuda:
        loss = loss.cuda()
    target = target.view(-1, 1)
    return loss(prediction, target)

def jsgan_d_fake_loss(d_fake):
    return bce(d_fake, 0)

def jsgan_d_real_loss(d_real):
    return bce(d_real, 1)

def jsgan_g_loss(d_fake):
    return bce(d_fake, 1)

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
        self.g_loss = jsgan_g_loss
        self.d_loss_real = jsgan_d_real_loss
        self.d_loss_fake = jsgan_d_fake_loss
        if self.dnorm is None:
            self.dnorm = 0.

    def __str__(self):
        str_ = "gx:\n"
        str_ += str(self.gx)
        str_ += "gz:\n"
        str_ += str(self.gz)
        str_ += "dx:\n"
        str_ += str(self.dx)
        str_ += "dxz:\n"
        str_ += str(self.dxz)
        return str_
        
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
                rnd_state.normal(0, 1, size=(bs, self.z_dim))
            ).float()
        else:
            z = torch.from_numpy(
                np.random.normal(0, 1, size=(bs, self.z_dim))
            ).float()
        if self.use_cuda:
            z = z.cuda()
        return z
    
    def grad_norm(self, d_out, x):
        grad_wrt_x = grad(outputs=d_out, inputs=x,
                          grad_outputs=torch.ones(d_out.size()).cuda(),
                          create_graph=True, retain_graph=True, only_inputs=True)[0]
        g_norm = (grad_wrt_x.view(
            grad_wrt_x.size()[0], -1).norm(2, 1)**2).mean()
        return g_norm
    
    def train_on_instance(self, z, x, **kwargs):
        self._train()
        # Zero gradients.
        self.dx.zero_grad()
        self.dxz.zero_grad()
        self.gx.zero_grad()
        self.gz.zero_grad()
        batch_size = x.size(0)
        if self.use_cuda:
            z = z.cuda()
        x.requires_grad = True
        x_fake = self.gx(z) # z -> x'
        encoded = self.gz(x) # x -> z'
        # reparameterisation trick
        eps = self.sample_z(batch_size)
        eps = eps.view(-1, eps.size(1), 1, 1)
        eps.require_grad = False
        z_enc = encoded[:, :self.z_dim] + \
                encoded[:, self.z_dim:].exp() * eps
        x_recon = self.gx(z_enc.view(-1, z_enc.size(1)))
        if self.lamb > 0:
            recon = torch.mean((x_recon - x)**2)
        # Update the generator.
        dx_true = self.dx(x)
        dx_fake = self.dx(x_fake)
        d_true = self.dxz(torch.cat((dx_true, z_enc), dim=1)) # (x, z')
        d_fake = self.dxz(torch.cat((dx_fake, z.view(-1, z.size(1), 1, 1)), dim=1)) # (x', z)
        loss_g = self.d_loss_real(d_fake) + self.d_loss_fake(d_true)
        if self.lamb > 0:
            loss_g += self.lamb*recon
        loss_g.backward()
        self.optim['g'].step()
        # Update the discriminator.
        self.dx.zero_grad()
        self.dxz.zero_grad()
        dx_true = self.dx(x)
        dx_fake = self.dx(x_fake.detach())
        # Make sure to detach z_enc and x_fake
        d_true = self.dxz(torch.cat((dx_true, z_enc.detach()),
                                    dim=1)) # (x, z')
        d_fake = self.dxz(torch.cat((dx_fake, z.view(-1, z.size(1), 1, 1)),
                                    dim=1)) # (x', z)
        loss_d = self.d_loss_real(d_true) + self.d_loss_fake(d_fake)
        loss_d.backward(retain_graph=True)
        self.optim['d'].step()
        # Do gradient penalty regularisation
        g_norm_x = self.grad_norm(d_true, x)
        if self.dnorm > 0.:
            self.optim['d'].zero_grad()
            (g_norm_x*self.dnorm).backward()
            self.optim['d'].step()
        losses = {
            'loss_d': loss_d.item(),
            'loss_g': loss_g.item()
        }
        if self.lamb > 0:
            losses['recon'] = recon.item()
        if self.dnorm > 0:
            losses['d_real_norm'] = g_norm_x.item()
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
