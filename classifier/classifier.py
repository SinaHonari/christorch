from .base import BaseModel
import torch
from torch import nn
from torch import optim
from torch.autograd import Variable

class Classifier(BaseModel):
    def __init__(self,
                 net,
                 num_classes,
                 opt=optim.Adam, opt_args={},
                 l2_decay=0.,
                 metrics=[],
                 handlers=[],
                 use_cuda='detect',
                 verbose=True):
        assert use_cuda in [True, False, 'detect']
        if use_cuda == 'detect':
            use_cuda = True if torch.cuda.is_available() else False
        self.net = net
        self.num_classes = num_classes
        self.verbose = verbose
        params = filter(lambda x: x.requires_grad, self.net.parameters())
        self.optim = opt(params, weight_decay=l2_decay, **opt_args)
        self.metrics = metrics
        self.handlers = handlers
        self.use_cuda = use_cuda
        if self.use_cuda:
            self.net.cuda()
        self.scheduler = []

    def prepare_batch(self, X, y):
        X, y = X.float(), y.long()
        if self.use_cuda:
            X, y = X.cuda(), y.cuda()
        X, y = Variable(X), Variable(y)
        return X, y

    def train_on_instance(self, X_batch, y_batch):
        out = self.net(X_batch)
        pdist = torch.exp(out)
        loss = nn.NLLLoss()(out, y_batch[:,0])
        self.optim.zero_grad()
        loss.backward()
        self.optim.step()
        # Compute accuracy.
        preds_np = pdist.data.cpu().numpy().argmax(axis=1)
        gt_np = y_batch.data.cpu().numpy()
        acc = (preds_np == gt_np).mean()
        return {
            'loss': loss.data.item(),
            'acc': acc
        }, {}

    def eval_on_instance(self, X_batch, y_batch):
        with torch.no_grad():
            out = self.net(X_batch)
            pdist = torch.exp(out)
            loss = nn.NLLLoss()(out, y_batch[:,0])
            preds_np = pdist.data.cpu().numpy().argmax(axis=1)
            gt_np = y_batch.data.cpu().numpy()
            acc = (preds_np == gt_np).mean()
        return {
            'loss': loss.data.item(),
            'acc': acc
        }, {}
