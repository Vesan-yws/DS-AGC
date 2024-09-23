import torch
import torch.nn as nn


class AutomaticWeightedLoss(nn.Module):
    """automatically weighted multi-task loss
    Params：
        num: int，the number of loss
        x: multi-task loss
    Examples：
        loss1=1
        loss2=2
        awl = AutomaticWeightedLoss(2)
        loss_sum = awl(loss1, loss2)
    """
    def __init__(self, num=2):
        super(AutomaticWeightedLoss, self).__init__()
        params = torch.ones(num, requires_grad=True)
        self.params = torch.nn.Parameter(params)
        print(self.params)

    def forward(self, *x):
        loss_sum = 0
        length = len(x)-1
        for i, loss in enumerate(x):
            loss_sum += 1 / (self.params[i] ** 2) * loss + torch.log(self.params[i])
            # if i == length:
            #     loss_sum += 1 / (self.params[i] ** 2) * loss + torch.log(self.params[i])
            # else:
            #     loss_sum += 0.5 / (self.params[i] ** 2) * loss + torch.log(self.params[i])
        return loss_sum

if __name__ == '__main__':
    awl = AutomaticWeightedLoss(4)
    awl(2.5,2.6,3.7,3.8)
    print(awl.parameters())
