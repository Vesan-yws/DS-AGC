# -*- coding: utf-8 -*-
"""
Created on Tue Dec 14 11:15:46 2021

@author: mindlab
"""
import torch.nn as nn
import torch
from torch.autograd import Function
import torch.nn.functional as F
from typing import Optional, Any, Tuple
import numpy as np

class ReverseLayerF(Function):

    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha
        return output, None

class GradientReverseFunction(Function):

    @staticmethod
    def forward(ctx: Any, input: torch.Tensor, coeff: Optional[float] = 1.) -> torch.Tensor:
        ctx.coeff = coeff
        output = input * 1.0
        return output

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> Tuple[torch.Tensor, Any]:
        return grad_output.neg() * ctx.coeff, None

class WarmStartGradientReverseLayer(nn.Module):
    """Gradient Reverse Layer :math:`\mathcal{R}(x)` with warm start

        The forward and backward behaviours are:

        .. math::
            \mathcal{R}(x) = x,

            \dfrac{ d\mathcal{R}} {dx} = - \lambda I.

        :math:`\lambda` is initiated at :math:`lo` and is gradually changed to :math:`hi` using the following schedule:

        .. math::
            \lambda = \dfrac{2(hi-lo)}{1+\exp(- α \dfrac{i}{N})} - (hi-lo) + lo

        where :math:`i` is the iteration step.

        Parameters:
            - **alpha** (float, optional): :math:`α`. Default: 1.0
            - **lo** (float, optional): Initial value of :math:`\lambda`. Default: 0.0
            - **hi** (float, optional): Final value of :math:`\lambda`. Default: 1.0
            - **max_iters** (int, optional): :math:`N`. Default: 1000
            - **auto_step** (bool, optional): If True, increase :math:`i` each time `forward` is called.
              Otherwise use function `step` to increase :math:`i`. Default: False
        """

    def __init__(self, alpha: Optional[float] = 1.0, lo: Optional[float] = 0.0, hi: Optional[float] = 1.,
                 max_iters: Optional[int] = 1000., auto_step: Optional[bool] = False):
        super(WarmStartGradientReverseLayer, self).__init__()
        self.alpha = alpha
        self.lo = lo
        self.hi = hi
        self.iter_num = 0
        self.max_iters = max_iters
        self.auto_step = auto_step

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """"""
        coeff = float(
            2.0 * (self.hi - self.lo) / (1.0 + np.exp(-self.alpha * self.iter_num / self.max_iters))
            - (self.hi - self.lo) + self.lo
        )
        if self.auto_step:
            self.step()
        return GradientReverseFunction.apply(input, coeff)

    def step(self):
        """Increase iteration number :math:`i` by 1"""
        self.iter_num += 1

def binary_accuracy(output: torch.Tensor, target: torch.Tensor) -> float:
    """Computes the accuracy for binary classification"""
    with torch.no_grad():
        batch_size = target.size(0)
        pred = (output >= 0.5).float().t().view(-1)
        correct = pred.eq(target.view(-1)).float().sum()
        correct.mul_(100. / batch_size)
        return correct

class Discriminator3(nn.Module):
    def __init__(self,hidden_1):
        super(Discriminator3,self).__init__()
        self.fc1=nn.Linear(hidden_1,hidden_1)
        # self.fc2=nn.Linear(hidden_1, 1)
        self.fc2 = nn.Linear(hidden_1, 3)
        self.dropout1 = nn.Dropout(p=0.25)
        self.sigmoid = nn.Sigmoid()
          
    def forward(self,x):
        x=self.fc1(x)
        x=F.relu(x)
#         x=F.leaky_relu(x)
        x=self.dropout1(x)
        x=self.fc2(x)
        x=self.sigmoid(x)
        return x


class Discriminator(nn.Module):
    def __init__(self, hidden_1):
        super(Discriminator, self).__init__()
        self.fc1 = nn.Linear(hidden_1, hidden_1)
        # self.fc2=nn.Linear(hidden_1, 1)
        self.fc2 = nn.Linear(hidden_1, 1)
        self.dropout1 = nn.Dropout(p=0.25)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        #         x=F.leaky_relu(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return x

class TripleDomainAdversarialLoss(nn.Module):

    def __init__(self, hidden_1,reduction: Optional[str] = 'mean',max_iter=100):
        super(TripleDomainAdversarialLoss, self).__init__()
        self.grl = WarmStartGradientReverseLayer(alpha=1.0, lo=0., hi=1., max_iters=max_iter, auto_step=True)
        self.domain_discriminator = Discriminator3(hidden_1)
        self.domain_discriminator_accuracy = None
        self.crition=nn.CrossEntropyLoss()

    def forward(self,x):
        f = self.grl(x)
        d = self.domain_discriminator(f)
        source_num = int(len(x) / 3)
        d_label_s = torch.ones((source_num, 1)).to(x.device)
        d_label_t = torch.zeros((source_num, 1)).to(x.device)
        d_label_u = torch.ones((source_num, 1)).to(x.device)+1
        label=torch.cat((d_label_s,d_label_t,d_label_u)).squeeze().long()
#        self.domain_discriminator_accuracy = 0.5 * (binary_accuracy(d_s, d_label_s) + binary_accuracy(d_t, d_label_t))
        return self.crition(d, label)


class DomainAdversarialLoss(nn.Module):

    def __init__(self,hidden_1, reduction: Optional[str] = 'mean',max_iter=100):
        super(DomainAdversarialLoss, self).__init__()
        self.grl = WarmStartGradientReverseLayer(alpha=1.0, lo=0., hi=1., max_iters=max_iter, auto_step=True)
        self.domain_discriminator = Discriminator(hidden_1)
        self.bce = nn.BCELoss(reduction=reduction)
        self.domain_discriminator_accuracy = None

    def forward(self, x):
        f = self.grl(x)
        d = self.domain_discriminator(f)


        d_s, d_t = d.chunk(2, dim=0)
        d_label_s = torch.ones(len(d_s), 1).to(x.device)
        d_label_t = torch.zeros(len(d_t), 1).to(x.device)


        self.domain_discriminator_accuracy = 0.5 * (binary_accuracy(d_s, d_label_s) + binary_accuracy(d_t, d_label_t))
        return 0.5 * (self.bce(d_s, d_label_s) + self.bce(d_t, d_label_t))

class DomainAdversarialLoss_threeada(nn.Module):

    def __init__(self,hidden_1, reduction: Optional[str] = 'mean',max_iter=100):
        super(DomainAdversarialLoss_threeada, self).__init__()
        self.grl = WarmStartGradientReverseLayer(alpha=1.0, lo=0., hi=1., max_iters=max_iter, auto_step=True)
        self.domain_discriminator = Discriminator(hidden_1)
        self.bce = nn.BCELoss(reduction=reduction)
        self.domain_discriminator_accuracy = None

    def forward(self, x):
        f = self.grl(x)
        d = self.domain_discriminator(f)

        source_num = int(len(x) / 3)
        d_s = d[0:2*source_num, :]
        d_t = d[2*source_num:, :]
        d_label_s = torch.ones(2 * source_num, 1).to(x.device)
        d_label_t = torch.zeros(source_num, 1).to(x.device)


        self.domain_discriminator_accuracy = 0.5 * (binary_accuracy(d_s, d_label_s) + binary_accuracy(d_t, d_label_t))
        return 0.5 * (self.bce(d_s, d_label_s) + self.bce(d_t, d_label_t))