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
import h5py
import util
from keras.preprocessing.image import ImageDataGenerator #TODO

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

def get_wiki_iterators(batch_size):
    """
    NOTE: the OS environment variable H5_WIKI must be set!!!
    """
    def preproc(img):
        img = util.min_max_then_tanh(img, )
        # random crops can work within the preprocessor
        # so long as the original img size does not change,
        # e.g. if you crop a smaller chunk from 256px you must
        # upsize it back to 256px
        img = util.rnd_crop(img, data_format='channels_last')
        return img
    def minimal_preproc(img):
        img = util.min_max_then_tanh(img, )
        return img
    h5 = os.environ['H5_WIKI']
    channels_mode = 'channels_last'
    dataset = h5py.File(h5,"r")
    imgen = ImageDataGenerator(horizontal_flip=True, data_format=channels_mode, preprocessing_function=preproc)
    it_train = util.ClassifierIterator(X_arr=dataset['xt'], y_arr=dataset['yt'],
                                bs=batch_size, imgen=imgen, mode='old',
                                rnd_state=np.random.RandomState(0),
                                data_format=channels_mode)
    it_val = util.ClassifierIterator(X_arr=dataset['xv'], y_arr=dataset['yv'],
                                bs=batch_size, imgen=imgen, mode='old',
                                rnd_state=np.random.RandomState(0),
                                data_format=channels_mode)
    return it_train, it_val

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
                 l2_decay=0.,
                 metrics={},
                 hooks={},
                 gpu_mode=False,
                 verbose=True):
        """
        loss_name: either 'x-ent' (cross-entropy) or 'emd2' (squared earth
          mover's distance). The latter is only appropriate if
          ordinal classification is being performed.
        metrics: a dict of the form {metric_name: metric_fn}, where
          `metric_fn` is a function that takes a minibatch of pdists
          (bs, k) and a minibatch of ground truths (k,) and returns
          some sort of metric, expressed as as a scalar.
        hooks: a dict of the form {hook_name: hook_fn}, where
          `hook_fn` is a function that takes (X_batch, y_batch, epoch)
          and gets called every minibatch, and performs some function.
          This can be useful for spitting out auxiliary information
          somewhere during training.
        """
        assert loss_name in ['x-ent', 'emd2']
        self.in_shp = in_shp
        self.num_classes = num_classes
        self.loss_name = loss_name
        self.verbose = verbose
        self.l_out = net_fn(in_shp, num_classes, **net_fn_params)
        self.optim = opt(self.l_out.parameters(), weight_decay=l2_decay, **opt_args)
        self.metrics = metrics
        self.hooks = hooks
        self.loss = F.nll_loss
        self.gpu_mode = gpu_mode
        if self.loss_name == 'emd2':
            from architectures.extensions import CMF
            self.l_pmf = CMF(self.num_classes)
        else:
            self.l_pmf = None
        if self.gpu_mode:
            self.l_out.cuda()
            if self.l_pmf != None:
                self.l_pmf.cuda()
            
    def train(self, itr_train, itr_valid, epochs, model_dir, result_dir, resume=False, max_iters=None, save_every=1, verbose=True):
        """
        If a results directory is specified, this will:
        - Save a CSV file (results.txt) with per-epoch statistics for training/validation.
        - Save a CSv file (valid_preds.txt) with predicted prob. dists on the validation set.
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
        stats_keys = ['epoch', 'train_loss', 'valid_loss', 'time']
        for epoch in range(epochs):
            ####
            stats = OrderedDict({})
            for key in stats_keys:
                stats[key] = None
            for key in self.metrics.keys():
                stats["%s_%s" % ('train', key)] = None
                stats["%s_%s" % ('valid', key)] = None
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
                num_batches = itr.N // itr.bs
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
                    if mode == 'train':
                        self.optim.zero_grad()
                    # compute output of network
                    out = F.log_softmax(self.l_out(X_batch))
                    pdist = torch.exp(out)
                    if self.loss == 'x-ent':
                        loss = self.loss(out, y_batch)
                    else:
                        # TODO: clean this up. maybe move the loss
                        # computation to another method
                        # create a one-hot y_batch as well
                        y_batch_onehot = np.zeros((len(y_batch), self.num_classes), dtype="float32")
                        y_batch_onehot[ np.arange(0, len(y_batch)), y_batch_orig ] = 1.
                        y_batch_onehot = torch.from_numpy(y_batch_onehot).float()
                        if self.gpu_mode:
                            y_batch_onehot = y_batch_onehot.cuda()
                        y_batch_onehot = Variable(y_batch_onehot)
                        #import pdb
                        #pdb.set_trace()
                        cmf_pdist = self.l_pmf(pdist)
                        cmf_y = self.l_pmf(y_batch_onehot)
                        # compute the squared L2 norm
                        loss = torch.mean(torch.sum((cmf_pdist-cmf_y)**2,dim=1))
                    tmp_stats['%s_loss' % mode].append(loss.data[0])
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
                stats['%s_loss' % mode] = np.mean(tmp_stats['%s_loss' % mode])
                stats['time'] = time.time() - epoch_start_time
                # save validation preds to disk
                if mode == "valid" and (epoch+1) % save_every == 0:
                    preds_matrix = np.hstack((all_pdist, all_ys[np.newaxis].T))
                    np.savetxt("%s/%s_preds_%i.csv" % (result_dir, mode, epoch+1), preds_matrix, delimiter=",")
            str_ = ",".join([ str(stats[key]) for key in stats ])
            print str_
            if f != None:
                f.write(str_ + "\n")
                f.flush()
            if self.gpu_mode and epoch == 1:
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
            #self.visualize_results("%s/%i.png" % (result_dir, epoch+1), sample_num=sample_num)
            #self.visualize_results("%s/f_%i.png" % (result_dir, epoch+1), sample_num=sample_num, fix=True)
            if (epoch+1) % save_every == 0 and model_dir != None:
                self.save( filename="%s/%i.pkl" % (model_dir, epoch+1) )
        if f != None:
            f.close()
        #utils.generate_animation(self.result_dir + '/',
        #                         epoch)
        #utils.loss_plot(train_hist,
        #                os.path.join(self.model_dir), self.model_name)
    def save(self, filename):
        torch.save(self.l_out.state_dict(), filename)
    def load(self, filename, cpu=False):
        if cpu:
            map_location = lambda storage, loc: storage
        else:
            map_location = None
        self.l_out.load_state_dict(torch.load(filename, map_location=map_location))

if __name__ == '__main__':

    def test(mode):
        assert mode in ['train', 'test']
        from architectures import resnet
        from metrics import acc, acc_exp, mae, mae_exp, lwk
        from hooks import get_dump_images
        cls = Classifier(
            net_fn=resnet.resnet18,
            net_fn_params={},
            in_shp=256, num_classes=101,
            metrics=OrderedDict({'acc':acc, 'acc_exp': acc_exp, 'mae':mae, 'mae_exp': mae_exp, 'lwk':lwk}),
            hooks={'dump_images': get_dump_images(5, "tmp")},
            gpu_mode=True)
        print cls
        name = "test"
        itr_train, itr_valid = get_wiki_iterators(196)
        if mode == "train":
            #cls.load("models/%s/10.pkl.bak2" % name)
            cls.train(itr_train=itr_train,
                      itr_valid=itr_valid,
                      epochs=100,
                      model_dir="models/%s" % name,
                      result_dir="results/%s" % name,
                      save_every=10, resume=True)
        elif mode == "test":
            cls.load("models/%s/20.pkl.bak3" % name)
            import pdb
            pdb.set_trace()

    def test_emd2(mode):
        assert mode in ['train', 'test']
        from architectures import resnet
        from metrics import acc, acc_exp, mae, mae_exp, lwk
        from hooks import get_dump_images
        cls = Classifier(
            net_fn=resnet.resnet18,
            net_fn_params={},
            in_shp=256, num_classes=101,
            loss_name='emd2',
            metrics=OrderedDict({'acc':acc, 'acc_exp': acc_exp, 'mae':mae, 'mae_exp': mae_exp, 'lwk':lwk}),
            hooks={'dump_images': get_dump_images(5, "tmp")},
            gpu_mode=True)
        print cls
        name = "test_emd2.s2"
        itr_train, itr_valid = get_wiki_iterators(196)
        if mode == "train":
            cls.train(itr_train=itr_train,
                      itr_valid=itr_valid,
                      epochs=100,
                      model_dir="models/%s" % name,
                      result_dir="results/%s" % name,
                      save_every=5, resume=True)


            
    locals()[sys.argv[1]]( sys.argv[2] )
