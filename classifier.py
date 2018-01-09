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

class FakeIterator():
    def __init__(self, inp_shape, num_classes):
        self.inp_shape = inp_shape
        self.num_classes = num_classes
        self.fn = self._iterate()
        self.N = 100
        self.bs = inp_shape[0]
    def _iterate(self):
        while True:
            X_batch = np.random.normal(0, 1, size=self.inp_shape).astype("float32")
            y_batch = np.random.randint(0, self.num_classes, size=(self.inp_shape[0],)).astype("int32")
            yield X_batch, y_batch
    def __iter__(self):
        return self
    def next(self):
        return self.fn.next()

from memory_profiler import profile
    
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
                 opt=optim.Adam, opt_args={'lr':1e-3, 'betas':(0.9, 0.999)},
                 scheduler=None,
                 scheduler_args={},
                 scheduler_metric='valid_loss',
                 l2_decay=0.,
                 metrics={},
                 hooks={},
                 gpu_mode='detect',
                 verbose=True):
        """
        loss_name: either 'x-ent' (cross-entropy), 'emd2' (squared earth
          mover's distance), or 'bce' (binary cross-entropy).
        metrics: a dict of the form {metric_name: metric_fn}, where
          `metric_fn` is a function that takes a minibatch of pdists
          (bs, k) and a minibatch of ground truths (k,) and returns
          some sort of metric, expressed as as a scalar.
        scheduler:
        scheduler_args:
        scheduler_metric:
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
        # https://github.com/pytorch/pytorch/issues/1266
        params = filter(lambda x: x.requires_grad, self.l_out.parameters())
        #params = self.l_out.parameters()
        self.optim = opt(params, weight_decay=l2_decay, **opt_args)
        if scheduler != None:
            self.scheduler = scheduler(self.optim, **scheduler_args)
        else:
            self.scheduler = None
        self.scheduler_metric = scheduler_metric
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

    def train(self, itr_train, itr_valid, epochs, model_dir, result_dir, resume=False, max_iters=None, save_every=1, verbose=True):
        """
        If a results directory is specified, this will:
        - Save a CSV file (results.txt) with per-epoch statistics for training/validation.
        - Save a CSV file (valid_preds.txt) with predicted prob. dists on the validation set.
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
        f = open("%s/results.txt" % result_dir, "wb" if not resume else "a") if result_dir != None else None
        start_time = time.time()
        stats_keys = ['epoch', 'train_loss', 'valid_loss', 'lr', 'time']
        for epoch in range(epochs):
            ####
            stats = OrderedDict({})
            for key in stats_keys:
                stats[key] = None
            for key in self.metrics.keys():
                stats["%s_%s" % ('train', key)] = None
                stats["%s_%s" % ('valid', key)] = None
                '''
                if self.loss_name == 'bce':
                    # add a new way of doing prediction
                    stats["%s_%s_bin" % ('train', key)] = None
                    stats["%s_%s_bin" % ('valid', key)] = None
                '''
            ####
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
                num_batches = int(math.ceil(itr.N / itr.bs))
                all_ys, all_pdist = [], [] # accumulate labels, pdists
                for b in tqdm(range(num_batches)):
                    X_batch, y_batch = itr.next()
                    y_batch_orig = y_batch
                    for key in self.hooks.keys():
                        self.hooks[key](X_batch, y_batch, epoch+1)
                    X_batch, y_batch = torch.from_numpy(X_batch).float(), torch.from_numpy(y_batch).long()
                    if self.gpu_mode:
                        X_batch = X_batch.cuda()
                        y_batch = y_batch.cuda()
                    X_batch, y_batch = Variable(X_batch), Variable(y_batch)
                    # clear gradients
                    self.optim.zero_grad()
                    # compute output of network
                    # TODO: refactor: --out--, pdist, loss = compute(...)
                    if self.loss_name == 'x-ent':
                        out = self.l_out(X_batch)
                        pdist = torch.exp(out)
                        loss = nn.NLLLoss()(out, y_batch)
                    elif self.loss_name == 'emd2':
                        out = self.l_out(X_batch)
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
                    else:
                        y_batch_cum = torch.from_numpy(util.int_to_ord(y_batch_orig, self.num_classes)).float()
                        if self.gpu_mode:
                            y_batch_cum = y_batch_cum.cuda()
                        y_batch_cum = Variable(y_batch_cum)
                        out = self.l_out(X_batch)
                        loss = nn.BCEWithLogitsLoss()(out, y_batch_cum)
                        pdist = self.l_ctd(torch.exp(out))
                    tmp_stats['%s_loss' % mode].append(loss.data[0])
                    # after this part, it does the mem error here...
                    if mode == 'train':
                        loss.backward()
                        self.optim.step()
                    all_ys = np.hstack((all_ys, y_batch.cpu().data.numpy()))
                    if all_pdist == []:
                        all_pdist = pdist.cpu().data.numpy()
                    else:
                        all_pdist = np.vstack((all_pdist, pdist.cpu().data.numpy()))
                for key in self.metrics:
                    stats["%s_%s" % (mode, key)] = self.metrics[key](all_pdist, all_ys, self.num_classes)
                '''
                if self.loss_name == 'bce':
                    ########################################################################
                    # TODO: find a nicer way to do this, since we've hardcoded
                    # the biases key name, and also performed the square. Maybe
                    # we can expose the state dict in the metrics??
                    # if we're using ordered logit, then also compute expectation bin score
                    ########################################################################
                    biases = self.l_out.state_dict()['classifier.biases'].cpu().numpy()**2
                    c_biases = [ sum(biases[0,0:k+1]) for k in range(len(biases[0])) ]
                    c_biases = [0.] + c_biases
                    all_pdist_bin = np.eye(self.num_classes)[ self.bin_expectation(c_biases, all_pdist) ]
                    for key in self.metrics:
                        stats['%s_%s_bin' % (mode, key)] = self.metrics[key](all_pdist_bin, all_ys, self.num_classes)
                '''
                stats['%s_loss' % mode] = np.mean(tmp_stats['%s_loss' % mode])
                # save validation preds to disk
                if mode == "valid" and (epoch+1) % save_every == 0:
                    preds_matrix = np.hstack((all_pdist, all_ys[np.newaxis].T))
                    np.savetxt("%s/%s_preds_%i.csv" % (result_dir, mode, epoch+1), preds_matrix, delimiter=",")
            stats['lr'] = self.optim.state_dict()['param_groups'][0]['lr']
            stats['time'] = time.time() - epoch_start_time
            # update the learning rate scheduler, if applicable
            if self.scheduler != None:
                self.scheduler.step(stats[self.scheduler_metric])
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
    def bin_expectation(self, bins, ps):
        """
        This is a bit of a generalisation of the expectation trick
          (aka softmax ordinal expected value) seen in multiple
          papers, including my own. Normally, in an ordinal problem,
          we can compute the expectation by calculating the dot
          product between [0, ..., K-1] and p(y|x), but in this case
          we replace [0, ..., K-1] with [0, a1, ..., a_{k-1}], where
          the sequence is sorted and 0 <= a1 <= ... <= a_{k-1}.
        bins: Sorted bins [0, a1, a2, ..., ak] of length K (number
          of classes)
        ps: probability distributions
        """
        idxs = []
        for i in range(len(ps)):
            fake_pred = np.dot(ps[i], bins)
            # basically, figure out which bin the prediction is in
            pred_idx = np.sum( fake_pred >= bins ) - 1
            idxs.append(pred_idx)
        return np.asarray(idxs)

