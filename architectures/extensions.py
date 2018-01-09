from torchvision.models import resnet
from torchvision.models.resnet import ResNet, BasicBlock
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import math
import numpy as np

class CMF(nn.Module):
    """
    A dense layer which has an lower-triangular weight matrix of ones
    to implement conversion of a PMF to a CMF.
    ------
    Notes:
    ------
    - https://discuss.pytorch.org/t/what-is-the-recommended-way-to-re-assign-update-values-in-a-variable-or-tensor/6125/3
    """
    def __init__(self, num_classes):
        super(CMF, self).__init__()
        self.uro = nn.Linear(num_classes, num_classes, bias=False)
        # make the matrix all-ones
        self.uro.weight.data *= 0.
        for k in range(0, num_classes):
            self.uro.weight.data[k,0:(k+1)] += 1.
        self.uro.weight.requires_grad = False
        #print self.uro.weight.data
        #print W
    def forward(self, x):
        return self.uro(x)

class BinomialExtension(nn.Module):
    """
    Module that implements the log PMF of the binomial distribution
    normalised by the softmax nonlinearity.
    num_in: number of units of the previous layer
    num_classes: number of classes
    learn_tau: either 'none', 'bias', or 'function'. 'none' means
      we don't use it, 'bias' means we learn it as a bias (which
      will mean the same tau is applied to any example), and
      'function' means that we will compute a tau for each
      example.
    ------
    Notes:
    ------
    """
    def __init__(self, num_in, num_classes, learn_tau='none'):
        assert learn_tau in ['none', 'bias', 'function']
        from scipy.special import binom
        super(BinomialExtension, self).__init__()
        k = num_classes
        # layers
        self.l_sigm = nn.Linear(num_in, 1)
        self.l_copy = nn.Linear(1, k, bias=False)
        self.l_copy.weight.data *= 0
        self.l_copy.weight.data += 1
        self.l_copy.weight.requires_grad = False
        # variables
        c = np.arange(0, k)
        self.c = torch.from_numpy(c.reshape((1, k))).float()
        self.binom_coefs = torch.from_numpy(binom(k-1, c)).float()
        if learn_tau == 'bias':
            self.tau = nn.Parameter(
                torch.from_numpy(np.asarray([0])).float()
            )
        elif learn_tau == 'function':
            self.tau_fn = nn.Linear(num_in, 1)
        self.learn_tau = learn_tau
        self.num_classes = num_classes

    def forward(self, x):
        inp = x
        k = self.num_classes
        eps = 1e-4
        x = F.sigmoid(self.l_sigm(x))
        x = self.l_copy(x)
        if x.is_cuda:
            c = Variable(self.c).cuda()
            binom_coefs = Variable(self.binom_coefs).cuda()
        else:
            c = Variable(self.c)
            binom_coefs = Variable(self.binom_coefs)
        x = torch.log(binom_coefs) + \
            c*torch.log(x+eps) + \
            (k-1-c)*torch.log(1.-x+eps)
        if self.learn_tau == 'bias':
            x = x / F.sigmoid(self.tau)
        elif self.learn_tau == 'function':
            x = x / F.sigmoid(self.tau_fn(inp))
        return x

'''
class CumulativeToDiscrete(nn.Module):
    """
    Given a cumulative probability distribution layer whose
      units are p(j >= 0), ..., p(j >= K-2) (idx from 0), create 
      a layer which subtracts adjacent cumulative probabilities
      to obtain discrete probabilities p(j = 1), ..., p(j = K).
    We can illustrate this diagramatically, and supposing we have
      cumulative probs p(y > 0), ..., p(y > 3) (and we have 5 classes):
      p(y = 0) = 1 - p(y > 0)
      p(y = 1) = p(y > 1) - p(y > 0)
      p(y = 2) = p(y > 2) - p(y > 1)
      p(y = 3) = p(y > 3) - p(y > 2)
      p(y = 4) = p(y > 3)
    """
    def __init__(self, num_inputs):
        l_sub = nn.Linear(num_inputs, num_inputs+1)
        l_sub.weight.data[0,0]=1
        for k in range(1, num_inputs):
            l_sub.weight.data[k-1,k] = -1
            l_sub.weight.data[k,k] = 1
        l_sub.weight.data[num_inputs-1,num_inputs] = 1
        l_sub.bias.data *= 0.
        l_sub.bias.data[0, num_inputs] = 1
        self.l_sub = l_sub
    def forward(self, x):
        x = self.l_sub(x)
        #x = torch.abs(
'''


"""
h(y <= j | x) = g( f(x) + b_j ) (can g() just be linear?)
h(y = j | x ) = | h(y <= j | x ) - h(y <= j+1 | x )
                | edge cases...
Then, p(y = j | x) = softmax( h(y|x) )_{j}

m_biases = ELU(b) * U

"""

class POM(nn.Module):
    """
    Proportional odds model extension, which models cumulative
      probabilities p(y <= j | x) = sigm(b_j + f(x))

    Parameters
    ----------
    mode: if 'cumulative', the output units measure the cumulative
      probabilities p(y <= j). If 'discrete', the output units
      measure the discrete probabiltiies p(y = j). This is done
      by computing the following for 0 <= j <= K-1:
      - p(y = 0) = p(y <= j) for j = 0
      - p(y = j) = p(y <= j) - p(y <= j-1) for j = 1..K-2
      - p(j = K-1) = 1 - ( p(y = 0) + ... + p(y = K-2) ) for j=K-1
    nonlinearity: what nonlinearity should be applied to the scalar
      output f(x)? Options are 'linear' (no activation), and
      'min_relu', which is the opposite of ReLU, i.e., min(0,x).

    Notes
    -----
    You want to use the BCELoss with ordinal encodings for the labels. 
      E.g. for 5 classes we can use a 4-length encoding as follows,
      where the 4 units denote:
      [ p(y <= 0), p(y <= 1), p(y <= 2), p(y <= 3) ] (for y=0..4).
      From this definition, we can define the ground truth labels
      as the following:
      - class 0 = [1,1,1,1] (i.e., class 0 is <= to 0,1,2,3)
      - class 1 = [0,1,1,1] (i.e., class 1 is <= 1,2,3)
      - class 2 = [0,0,1,1] (i.e., class 2 is <= 2,3)
      - class 3 = [0,0,0,1] (i.e., class 3 is <= 3)
      - class 4 = [0,0,0,0] (i.e., class 4 is not <= to any)
    I am not sure whether the POM formulation is particularly good
      for when there are a large number of classes K. This is because
      the cumulative probability p( y <= j | x ) = sigm( f(x) + b_j ),
      and the sigmoid saturates far away from zero. Essentially, what
      this means is that subtracting cumulative probabilities will
      result in numerical imprecision since you are subtracting two
      quantities which are very close to each other.
    """
    def __init__(self, num_units, num_classes):
        super(POM, self).__init__()
        num_classes = num_classes-1
        self.l_fx = nn.Linear(num_units, 1, bias=False)
        self.l_copy = nn.Linear(1, num_classes, bias=False)
        self.l_copy.weight.data *= 0
        self.l_copy.weight.data += 1
        self.l_copy.weight.requires_grad = False
        # parameters / constants
        self.biases = nn.Parameter(
            #torch.from_numpy( np.random.normal(0,sd_bias,size=(1,num_classes)) ).float()
            torch.from_numpy( np.linspace(-1,-1,num_classes)[np.newaxis] ).float()
        )
        # upper right ones matrix, for making the
        # biases monotonic
        uro = np.zeros((num_classes, num_classes))
        for k in range(num_classes):
            uro[k,0:(k+1)] = 1.
        self.uro = Variable(torch.from_numpy(uro.T).float())
        self.forward_ran = False
    def forward(self, x):
        x = self.l_fx(x)
        x = self.l_copy(x)
        if x.is_cuda and not self.forward_ran:
            self.uro = self.uro.cuda()
            self.forward_ran = True 
        m_biases = torch.mm( self.biases**2, self.uro)
        x = F.sigmoid(x+m_biases)
        return x

class StickBreakingOrdinal(nn.Module):
    """
    Notes
    -----

    Personal communication with Alex Piche: the problem is that
     for small values of K the distribution will not sum to 1.
     This means that we either artificially increase the number
     of classes (meaning we truncate p(y|x) at test time), or
     we add an extra class which is one minus the total length
     of the stick. We opt for the latter here since it seems
     cleaner.

    Note: this has an interpretation of sequentially scanning
      through the classes and assigning probability space,
      which makes sense in an ordinal context.
    """
    def __init__(self, num_units, num_classes):
        super(StickBreakingOrdinal, self).__init__()
        k = num_classes
        self.l_eta = nn.Linear(num_units, k-1, bias=False)
        # accumulator
        self.cmat = Variable( torch.from_numpy(np.tri(k-1, k-1).T).float() )
        # helper mat/bias for
        # 1 - sum(rest of stick)
        self.smat = np.eye(k-1, k)
        self.smat[:,-1] -= 1
        self.smat = Variable( torch.from_numpy(self.smat).float() )
        self.sb = np.zeros((1,k))
        self.sb[0, -1] = 1.
        self.sb = Variable( torch.from_numpy(self.sb).float() )
        self.eps = 1e-4
        self.forward_ran = False
    def forward(self, x):
        if x.is_cuda and not self.forward_ran:
            self.cmat = self.cmat.cuda()
            self.smat = self.smat.cuda()
            self.sb = self.sb.cuda()
            self.forward_ran = True
        eta = self.l_eta(x)
        eta_accum = torch.mm( F.softplus(eta), self.cmat)
        v = torch.exp(eta - eta_accum)
        v = torch.mm(v, self.smat) + self.sb
        return v + self.eps
    
class CumulativeToDiscrete(nn.Module):
    """

    """
    def __init__(self, num_in):
        super(CumulativeToDiscrete,self).__init__()
        D = np.zeros((num_in, num_in+1))
        D[0,0]=1
        for k in range(1, num_in):
            D[k-1,k] = -1
            D[k,k] = 1
        D[num_in-1,num_in] = 1
        b = np.zeros((1, num_in+1))
        b[0, num_in] = 1
        self.D = Variable(torch.from_numpy(D).float())
        self.b = Variable(torch.from_numpy(b).float())
        self.eps = 1e-5
        self.forward_ran = False
    def forward(self, x):
        if x.is_cuda and not self.forward_ran:
            self.D = self.D.cuda()
            self.b = self.b.cuda()
            self.forward_ran = True
        #x += self.eps
        x = torch.mm(x, self.D)
        x = torch.abs(self.b - x)
        return x
        #return torch.log(x)

'''
class OrderedLogits(nn.Module):
    def __init__(self, num_units, num_classes):
        super(POM, self).__init__()
        num_classes = num_classes-1
        self.l_fx = nn.Linear(num_units, 1, bias=False)
        self.l_copy = nn.Linear(1, num_classes, bias=False)
        self.l_copy.weight.data *= 0
        self.l_copy.weight.data += 1
        self.l_copy.weight.requires_grad = False
        # parameters / constants
        self.biases = nn.Parameter(
            torch.from_numpy( np.random.normal(0,1,size=(1,num_classes)) ).float()
        )
        # upper right ones matrix, for making the
        # biases monotonic
        uro = np.zeros((num_classes, num_classes))
        for k in range(num_classes):
            uro[k,0:(k+1)] = 1.
        self.uro = Variable(torch.from_numpy(uro.T).float())
        D = np.zeros((num_classes, num_classes+1))
        D[0,0]=1
        for k in range(1, num_classes):
            D[k-1,k] = -1
            D[k,k] = 1
        D[num_in-1,num_in] = 1
        b = np.zeros((1, num_classes+1))
        b[0, num_in] = 1
        self.D = Variable(torch.from_numpy(D).float())
        self.b = Variable(torch.from_numpy(b).float())
        self.forward_ran = False
    def forward(self, x):
        x = self.l_fx(x)
        x = self.l_copy(x)
        if x.is_cuda and not self.forward_ran:
            self.D = self.D.cuda()
            self.b = self.b.cuda()
            self.uro = self.uro.cuda()
            self.forward_ran = True
        m_biases = torch.mm( self.biases**2, self.uro )
        x = F.relu(x+m_biases) # pseudo-cumulative probs
        x = torch.mm(x, self.D) # 
        return x
'''


        
'''
class OrdinalSubtractLayer(Layer):
    def __init__(self, incoming, **kwargs):
        super(OrdinalSubtractLayer, self).__init__(incoming, **kwargs)
        num_inputs = int(np.prod(self.input_shape[1:]))
        self.num_inputs = num_inputs
        # construct the matrix
        self.W = np.zeros((num_inputs, num_inputs+1), dtype="float32")
        self.W[0,0]=1
        for k in range(1, num_inputs):
            self.W[k-1,k] = -1
            self.W[k,k] = 1
        self.W[num_inputs-1,num_inputs] = 1
        # construct the bias row vector
        self.b = np.zeros((1, num_inputs+1), dtype="float32")
        self.b[0, num_inputs] = 1
        #print self.W
        #print self.b

    def get_output_shape_for(self, input_shape):
        return (input_shape[0], self.num_inputs+1)

    def get_output_for(self, input, **kwargs):
        if input.ndim > 2:
            # if the input has more than two dimensions, flatten it into a
            # batch of feature vectors.
            input = input.flatten(2)
        result = T.dot(input, self.W)
        result = T.abs_(self.b - result)
        #eps = 0.01
        #result = result + eps
        return result
'''


    
if __name__ == '__main__':
    from torch.autograd import Variable
    import numpy as np
    net = nn.Sequential(
        nn.Linear(10,5),
        StickBreakingOrdinal(5, 5),
    )
    
    x_fake = np.random.normal(0,1,size=(2,10))
    x_fake = Variable(torch.from_numpy(x_fake).float())
    out = net(x_fake)
    #loss = torch.mean(out)
    #loss.backward()
    #ctd = CumulativeToDiscrete(4)
    
    print net
    import pdb
    pdb.set_trace()
    

