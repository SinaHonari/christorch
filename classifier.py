import torch, time, os, pickle
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from collections import OrderedDict
import os
import sys
from tqdm import tqdm
from . import util
    
class Classifier():
    #def num_parameters(self):
    #    return np.sum([ np.prod(np.asarray(elem.size())) for elem in self.l_out.parameters() ])
    #def __str__(self):
    #    return str(self.l_out) + "\n# parameters:" + str(self.num_parameters())
    def __init__(self,
                 net_fn,
                 opt=optim.Adam, opt_args={},
                 l2_decay=0.,
                 metrics={},
                 hooks={},
                 handlers=[],
                 use_cuda='detect',
                 verbose=True):
        """
        loss_name: either 'x-ent' (cross-entropy), 'emd2' (squared earth
          mover's distance), or 'bce' (binary cross-entropy).
        metrics: a dict of the form {metric_name: metric_fn}, where
          `metric_fn` is a function that takes a minibatch of pdists
          (bs, k) and a minibatch of ground truths (k,) and returns
          some sort of metric, expressed as as a scalar.
        hooks: a dict of the form {hook_name: hook_fn}, where
          `hook_fn` is a function that takes (X_batch, y_batch, epoch)
          and gets called every minibatch, and performs some function.
          This can be useful for spitting out auxiliary information
          somewhere during training.
        gpu_mode: if `True`, we assume GPU mode is enabled. This can
          also be set to the string `autodetect`, which will automatically
          enable the mode if a GPU is detected.
        """
        assert use_cuda in [True, False, 'detect']
        if use_cuda == 'detect':
            use_cuda = True if torch.cuda.is_available() else False
        self.verbose = verbose
        self.l_out = net_fn
        params = filter(lambda x: x.requires_grad, self.l_out.parameters())
        self.optim = opt(params, weight_decay=l2_decay, **opt_args)
        self.metrics = metrics
        self.hooks = hooks
        self.handlers = handlers
        self.use_cuda = use_cuda
        self.loss = nn.NLLLoss()
        self.last_epoch = 0
        if self.use_cuda:
            self.l_out.cuda()
            self.loss.cuda()

    def prepare_batch(self, X_batch, y_batch):
        X_batch = X_batch.float()
        y_batch = y_batch.long()
        if self.use_cuda:
            X_batch = X_batch.cuda()
            y_batch = y_batch.cuda()
        return X_batch, y_batch

    def train_on_instance(self, X_batch, y_batch, **kwargs):
        self.optim.zero_grad()
        out = self.l_out(X_batch)
        with torch.no_grad():
            acc = (out.argmax(dim=1) == y_batch).float().mean()
        pdist = torch.exp(out)
        loss = self.loss(out, y_batch)
        loss.backward()
        self.optim.step()
        return {
            'loss': loss.item(),
            'acc': acc.item()
        }, {'pdist': pdist}

    def eval_on_instance(self, X_batch, y_batch, **kwargs):
        with torch.no_grad():
            out = self.l_out(X_batch)
            acc = (out.argmax(dim=1) == y_batch).float().mean()
            pdist = torch.exp(out)
            loss = self.loss(out, y_batch)
        return {
            'loss': loss.item(),
            'acc': acc.item()
        }, {'pdist': pdist}

    def _get_stats(self, dict_, mode):
        dd = OrderedDict({})
        for key in dict_:
            if key != 'epoch':
                dd[key] = np.mean(dict_[key])
        return dd
    
    def train(self,
              itr_train,
              itr_valid,
              epochs,
              model_dir=None,
              result_dir=None,
              append=False,
              save_every=1,
              verbose=True):
        for folder_name in [model_dir, result_dir]:
            if folder_name is not None and not os.path.exists(folder_name):
                os.makedirs(folder_name)
        f_mode = 'w' if not append else 'a'
        f = None
        if result_dir is not None:
            f = open("%s/results.txt" % result_dir, f_mode)
        for epoch in range(self.last_epoch, epochs):
            # Training
            epoch_start_time = time.time()
            if verbose:
                pbar = tqdm(total=len(itr_train))
            train_dict = OrderedDict({'epoch': epoch+1})
            for b, (X_batch, y_batch) in enumerate(itr_train):
                X_batch, y_batch = self.prepare_batch(X_batch, y_batch)
                losses, outputs = self.train_on_instance(X_batch, y_batch,
                                                         iter=b+1)
                for key in losses:
                    this_key = 'train_%s' % key
                    if this_key not in train_dict:
                        train_dict[this_key] = []
                    train_dict[this_key].append(losses[key])
                pbar.update(1)
                pbar.set_postfix(self._get_stats(train_dict, 'train'))
                # Process handlers.
                for handler_fn in self.handlers:
                    handler_fn(losses, (X_batch, y_batch), outputs,
                               {'epoch':epoch+1, 'iter':b+1, 'mode':'train'})
            if verbose:
                pbar.close()
                pbar = tqdm(total=len(itr_valid))
            # Validation.
            valid_dict = {}
            for b, (X_batch, y_batch) in enumerate(itr_valid):
                X_batch, y_batch = self.prepare_batch(X_batch, y_batch)
                losses, outputs = self.eval_on_instance(X_batch, y_batch,
                                                         iter=b+1)
                for key in losses:
                    this_key = 'valid_%s' % key
                    if this_key not in valid_dict:
                        valid_dict[this_key] = []
                    valid_dict[this_key].append(losses[key])
                pbar.update(1)
                pbar.set_postfix(self._get_stats(valid_dict, 'valid'))
                # Process handlers.
                for handler_fn in self.handlers:
                    handler_fn(losses, (X_batch, y_batch), outputs,
                               {'epoch':epoch+1, 'iter':b+1, 'mode':'valid'})
            if verbose:
                pbar.close()
            # Step learning rates.
            #for key in self.scheduler:
            #    self.scheduler[key].step()
            all_dict = train_dict
            all_dict.update(valid_dict)
            for key in all_dict:
                all_dict[key] = np.mean(all_dict[key])
            all_dict["lr_%s" % key] = \
                self.optim.state_dict()['param_groups'][0]['lr']
            all_dict['time'] = \
                time.time() - epoch_start_time
            str_ = ",".join([str(all_dict[key]) for key in all_dict])
            print(str_)
            if f is not None:
                if (epoch+1) == 1 and not append:
                    # If we're not resuming, then write the header.
                    f.write(",".join(all_dict.keys()) + "\n")
                f.write(str_ + "\n")
                f.flush()
            if (epoch+1) % save_every == 0 and model_dir is not None:
                self.save(filename="%s/%i.pkl" % (model_dir, epoch+1),
                          epoch=epoch+1)            
        if f is not None:
            f.close()

    def save(self, filename):
        torch.save(self.l_out.state_dict(), filename)

    def load(self, filename, cpu=False):
        if cpu:
            map_location = lambda storage, loc: storage
        else:
            map_location = None
        self.l_out.load_state_dict(torch.load(
            filename, map_location=map_location))
