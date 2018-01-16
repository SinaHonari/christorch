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
import math
from tqdm import tqdm
import h5py
import util
    
class Classifier():
    def num_parameters(self):
        return np.sum([ np.prod(np.asarray(elem.size())) for elem in self.l_out.parameters() ])
    def __str__(self):
        return str(self.l_out) + "\n# parameters:" + str(self.num_parameters())
    def __init__(self,
                 net_fn,
                 net_fn_params,
                 in_shp,
                 num_classes,
                 loss_name='x-ent',
                 opt=optim.Adam, opt_args={},
                 l2_decay=0.,
                 metrics={}, hooks={},
                 gpu_mode='detect',
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
        assert loss_name in ['x-ent', 'emd2', 'bce']
        assert gpu_mode in [True, False, 'detect']
        if gpu_mode == 'detect':
            gpu_mode = True if torch.cuda.is_available() else False
        self.in_shp = in_shp
        self.num_classes = num_classes
        self.loss_name = loss_name
        self.verbose = verbose
        self.l_out = net_fn(in_shp, num_classes, **net_fn_params)
        # l_out is the instantiated class
        # l_out.forward(x) returns a dict of outputs...
        # https://github.com/pytorch/pytorch/issues/1266
        params = filter(lambda x: x.requires_grad, self.l_out.parameters())
        self.optim = opt(params, weight_decay=l2_decay, **opt_args)
        self.metrics = metrics
        self.hooks = hooks
        self.gpu_mode = gpu_mode
        if self.loss_name == 'emd2':
            from architectures.extensions import CMF
            self.l_pmf = CMF(self.num_classes)
            if self.gpu_mode:
                self.l_pmf.cuda()
        elif self.loss_name == 'bce':
            from architectures.extensions import CumulativeToDiscrete
            self.l_ctd = CumulativeToDiscrete(self.num_classes-1)
            if self.gpu_mode:
                self.l_ctd.cuda()
        if self.gpu_mode:
            self.l_out.cuda()
    def compute_loss(self, out, y_batch):
        """
        TODO: cross-entropy has a `weights` parameter which can be used
        to ignore certain labels. This means if our y_batch consists of
        ?? labels we can ignore them in the loss calculation...!
        """
        y_batch_orig = y_batch
        if self.loss_name == 'x-ent':
            #out = out_fn(X_batch)
            pdist = torch.exp(out)
            loss = nn.NLLLoss()(out, y_batch)
        elif self.loss_name == 'emd2':
            #out = out_fn(X_batch)
            pdist = torch.exp(out)
            # TODO: clean this up. maybe move the loss
            # computation to another method
            # create a one-hot y_batch as well
            y_batch_onehot = np.zeros((len(y_batch), self.num_classes), dtype=y_batch_orig.dtype)
            y_batch_onehot[ np.arange(0, len(y_batch)), y_batch_orig ] = 1.
            y_batch_onehot = torch.from_numpy(y_batch_onehot).float()
            if self.gpu_mode:
                y_batch_onehot = y_batch_onehot.cuda()
            y_batch_onehot = Variable(y_batch_onehot)
            cmf_pdist = self.l_pmf(pdist)
            cmf_y = self.l_pmf(y_batch_onehot)
            # compute the squared L2 norm
            loss = torch.mean(torch.sum((cmf_pdist-cmf_y)**2,dim=1))
        elif self.loss_name == 'bce':
            y_batch_cum = torch.from_numpy(util.int_to_ord(y_batch_orig, self.num_classes)).float()
            if self.gpu_mode:
                y_batch_cum = y_batch_cum.cuda()
            y_batch_cum = Variable(y_batch_cum)
            #out = out_fn(X_batch)
            loss = nn.BCEWithLogitsLoss()(out, y_batch_cum)
            pdist = self.l_ctd(torch.exp(out))   
        return loss, pdist
    def train(self,
              itr_train, itr_valid,
              epochs, model_dir, result_dir,
              resume=False,
              max_iters=None,
              save_every=1,
              scheduler=None,
              scheduler_args={},
              scheduler_metric='valid_loss',
              verbose=True):
        """
        If a results directory is specified, this will:
        - Save a CSV file (results.txt) with per-epoch statistics for training/validation.
        - Save a CSV file (valid_preds.txt) with predicted prob. dists on the validation set.

        ----------
        Parameters
        ----------
        itr_train: iterator for training. It is assumed this loops infinitely, and has the two fields `N`
          (total number of instances) and `bs` (batch size).
        itr_valid: same as above, but for validatioh.
        epochs: number of epochs
        model_dir: directory for where model files should be stored
        result_dir: directory for where results files should be stored
        resume: if `True`, append to the results file rather than overwriting it
        save_every: save models every this number of epochs
        verbose:
        max_iters: for debugging purposes. Set the maximum number of minibatches per epoch. -1 = don't use.
        """
        for folder_name in [model_dir, result_dir]:
            if folder_name is not None and not os.path.exists(folder_name):
                os.makedirs(folder_name)
        if scheduler != None:
            scheduler = scheduler(self.optim, **scheduler_args)
        else:
            scheduler = None
        f = open("%s/results.txt" % result_dir, "wb" if not resume else "a") if result_dir != None else None
        start_time = time.time()
        stats_keys = ['epoch', 'train_loss', 'valid_loss', 'lr', 'time']
        for epoch in range(epochs):
            stats = OrderedDict({})
            for key in stats_keys:
                stats[key] = None
            for metric_key in self.metrics.keys():
                for y_key in self.l_out.keys:
                    stats["%s_%s_%s" % ('train', metric_key, y_key)] = None
                    stats["%s_%s_%s" % ('valid', metric_key, y_key)] = None
            stats['epoch'] = epoch+1
            if (epoch+1) == 1:
                # if this is the start of training, print out / save the header
                if f != None and not resume:
                    f.write(",".join(stats.keys()) + "\n")
                print ",".join(stats.keys())                
            epoch_start_time = time.time()
            tmp_stats = {'train_loss':[], 'valid_loss':[]}
            for idx, itr in enumerate([itr_train, itr_valid]):
                mode = 'train' if idx==0 else 'valid'
                if mode == 'train':
                    self.l_out.train()
                else:
                    self.l_out.eval()
                buf = {}
                for key in self.l_out.keys:
                    buf[key] = {'ys':[], 'pdist':[]}
                # the iterator is assumed to return data in the form
                # (X_batch, y_batch_1, y_batch_2, ...)
                pbar = tqdm(total=len(itr))
                for b, (X_batch,y_packet) in enumerate(itr):
                    pbar.update(1)
                    self.optim.zero_grad()
                    # if len(y_packet) == 1, we know it's simply one label set,
                    # but if it is a matrix (where each line corresponds to a set of labels),
                    # it should match the key signature
                    if y_packet.size()[1] != len(self.l_out.keys):
                        raise Exception("The number of ys returned by the iterator must match "
                                        + "the number of outputs (keys) of the network!!!")
                    #X_batch = torch.from_numpy(X_batch).float()
                    X_batch = X_batch.float()
                    if self.gpu_mode:
                        X_batch = X_batch.cuda()
                    X_batch = Variable(X_batch)
                    outs = self.l_out(X_batch)
                    tot_loss = 0.
                    for y_idx, y_key in enumerate(self.l_out.keys):
                        y_batch = y_packet[:,y_idx]
                        for key in self.hooks.keys():
                            self.hooks[key](X_batch, y_batch, epoch+1)
                        #y_batch = torch.from_numpy(y_batch).long()
                        y_batch = y_batch.long()
                        if self.gpu_mode:
                            y_batch = y_batch.cuda()
                        y_batch = Variable(y_batch)
                        this_loss, this_pdist = self.compute_loss(outs[y_key], y_batch)
                        tot_loss += this_loss
                        buf[y_key]['ys'] = np.hstack((buf[y_key]['ys'], y_batch.cpu().data.numpy()))
                        if buf[y_key]['pdist'] == []:
                            buf[y_key]['pdist'] = this_pdist.cpu().data.numpy()
                        else:
                            buf[y_key]['pdist'] = np.vstack((buf[y_key]['pdist'], this_pdist.cpu().data.numpy()))
                    if mode == 'train':
                        tot_loss.backward()
                        self.optim.step()
                    #########
                    tmp_stats['%s_loss' % mode].append(tot_loss.data[0])
                pbar.close()
                for y_key in buf.keys():
                    for key in self.metrics:
                        stats["%s_%s_%s" % (mode, key, y_key)] = self.metrics[key](buf[y_key]['pdist'], buf[y_key]['ys'], self.num_classes)
                #########
                stats['%s_loss' % mode] = np.mean(tmp_stats['%s_loss' % mode])
                #if mode == "valid" and (epoch+1) % save_every == 0:
                #    preds_matrix = np.hstack((all_pdist, all_ys[np.newaxis].T))
                #    np.savetxt("%s/%s_preds_%i.csv" % (result_dir, mode, epoch+1), preds_matrix, delimiter=",")
            stats['lr'] = self.optim.state_dict()['param_groups'][0]['lr']
            stats['time'] = time.time() - epoch_start_time
            # update the learning rate scheduler, if applicable
            if scheduler != None:
                scheduler.step(stats[scheduler_metric])
            str_ = ",".join([ str(stats[key]) for key in stats ])
            print str_
            if f != None:
                f.write(str_ + "\n")
                f.flush()
            if self.gpu_mode and (epoch+1) == 1:
                # when we're in the first epoch, print how much memory
                # was used (allocated?) by the GPU
                try:
                    from pynvml import nvmlInit, nvmlDeviceGetHandleByIndex, nvmlDeviceGetMemoryInfo
                    nvmlInit()
                    handle = nvmlDeviceGetHandleByIndex(0)
                    totalMemory = nvmlDeviceGetMemoryInfo(handle)
                    print "GPU memory used:", totalMemory.used / 1024. / 1024.
                except Exception as e:
                    print "Was unable to detect amount of GPU memory used"
                    print e
            if (epoch+1) % save_every == 0 and model_dir != None:
                self.save( filename="%s/%i.pkl" % (model_dir, epoch+1) )
        if f != None:
            f.close()
    def dump_preds(self, itr, filename):
        """
        Dump predictions to a CSV file, in the format:
        p1,p2,...,y, where y = ground truth.
        """
        num_batches = int(math.ceil(itr.N / itr.bs))
        all_ys, all_pdist = [], [] # accumulate labels, pdists
        for b in range(num_batches):
            X_batch, y_batch = itr.next()
            X_batch = torch.from_numpy(X_batch).float()
            if self.gpu_mode:
                X_batch = X_batch.cuda()
            X_batch = Variable(X_batch)
            pdist = F.softmax(self.l_out(X_batch))
            all_ys = np.hstack((all_ys, y_batch))
            if all_pdist == []:
                all_pdist = pdist.cpu().data.numpy()
            else:
                all_pdist = np.vstack((all_pdist, pdist.cpu().data.numpy()))
        preds_matrix = np.hstack((all_pdist, all_ys[np.newaxis].T))
        np.savetxt(filename, preds_matrix, delimiter=",", fmt='%.8f') # 8 dp        
    def save(self, filename):
        torch.save(self.l_out.state_dict(), filename)
    def load(self, filename, cpu=False):
        if cpu:
            map_location = lambda storage, loc: storage
        else:
            map_location = None
        self.l_out.load_state_dict(torch.load(filename, map_location=map_location))

if __name__ == '__main__':
    import util
    itr_train = util.DebugIterator(img_size=256, num_classes=10, bs=4, n_outs=2, N=10)
    itr_valid = util.DebugIterator(img_size=256, num_classes=10, bs=4, n_outs=2, N=10)
    from architectures import resnet
    from metrics import *
    r = '18'
    cls = Classifier(
        net_fn=resnet.ResNetTwoOutput,
        net_fn_params={},
        in_shp=256, num_classes=10,
        metrics=OrderedDict({'acc':acc, 'acc_exp': acc_exp, 'mae':mae, 'mae_exp': mae_exp, 'qwk':qwk}),
        opt_args={'lr':1e-3},
        gpu_mode='detect',
    )
    print cls
    name = "test"
    mode = "train"
    if mode == "train":
        cls.train(itr_train=itr_train,
                  itr_valid=itr_valid,
                  epochs=100,
                  model_dir="models/%s" % name,
                  result_dir="results/%s" % name,
                  save_every=10)
    
