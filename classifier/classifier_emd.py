import torch
import numpy as np
from .classifier import Classifier
from ..architectures.extensions import CMF


class ClassifierEMD(Classifier):
    def __init__(self, *args, **kwargs):
        super(ClassifierEMD, self).__init__(*args, **kwargs)
        self.pmf = CMF(self.num_classes)
        if self.use_cuda:
            self.pmf.cuda()

    def prepare_batch(self, X_batch, y_batch):
        # We need to convert y_batch to a one-hot
        # representation.
        N = len(y_batch)
        y_batch_hot = torch.FloatTensor(N, self.num_classes)
        y_batch_hot.zero_()
        # UGLY
        y_batch = y_batch.long()
        for i in range(N):
            if y_batch[i]:
                y_batch_hot[i, y_batch[i]] = 1.
        if self.use_cuda:
            y_batch_hot = y_batch_hot.cuda()
        return X_batch, y_batch_hot

    def train_on_instance(self, X_batch, y_batch):
        out = self.net(X_batch)
        pdist = torch.exp(out)
        cmf_pdist = self.pmf(pdist)
        cmf_y = self.pmf(y_batch)
        self.optim.zero_grad()
        loss = torch.mean(torch.sum((cmf_pdist-cmf_y)**2,dim=1))
        loss.backward()
        self.optim.step()
        return {
            'loss': loss.data.item()
        }, {}

    def eval_on_instance(self, X_batch, y_batch):
        with torch.no_grad():
            out = self.net(X_batch)
            pdist = torch.exp(out)
            cmf_pdist = self.pmf(pdist)
            cmf_y = self.pmf(y_batch)
            loss = torch.mean(torch.sum((cmf_pdist-cmf_y)**2,dim=1))
        return {
            'loss': loss.data.item()
        }, {}
