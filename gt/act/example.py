import torch
import torch.nn as nn

from functools import partial

from torch_geometric.graphgym.config import cfg
from torch_geometric.graphgym.register import register_act

# partial: if:f(x, y, z)，then partial ,g = partial(f, y=2)，g(x, z) = f(x, y=2, z)。

# relu = max(0,x)
# sigmoid = 1/(1+exp(-x))
# LeakyReLU = x if x > 0 else alpha * x (alpha is a constant， < 1)
# GELU =0.5∗x∗(1+Tanh( 2/π ^ 0.5 ∗(x+0.044715∗x^3)))


class SWISH(nn.Module):
    def __init__(self, inplace=False):
        super().__init__()
        self.inplace = inplace

    def forward(self, x):
        if self.inplace:
            x.mul_(torch.sigmoid(x))
            return x
        else:
            return x * torch.sigmoid(x)

# f(x) = x * sigmoid(x)
# if inplace == True，then x = x * sigmoid(x)
# if inplace == False，then x1 =（x * sigmoid(x))


class SignedSqrt(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        x = torch.sqrt(torch.relu(x)) - torch.sqrt(torch.relu(-x))
        return x

# SignedSqrt f(x) = sqrt(max(0,x)) - sqrt(max(0,-x))
# torch.sqrt(torch.relu(x)):  ReLU  (torch.relu) : x ={0 if x < 0, x if x > 0}
# then sqrt(x) = {0 if x < 0, sqrt(x) if x > 0}
# torch.sqrt(torch.relu(-x)):


register_act('swish', partial(SWISH, inplace=True)) #cfg.mem.inplace
register_act('lrelu_03', partial(nn.LeakyReLU, negative_slope=0.3, inplace=True))
register_act('lrelu_02', partial(nn.LeakyReLU, negative_slope=0.2, inplace=True))
# Add Gaussian Error Linear Unit (GELU).
register_act('gelu', nn.GELU)
register_act('signed_sqrt', SignedSqrt)


# the raw graphgym，act.py（..\site-packages\torch_geometric\graphgym\models\act.py）：
'''
relu
selu
prelu
elu
lrelu_01 negative_slope=0.1
lrelu_025 negative_slope=0.25
lrelu_05 negative_slope=0.5

total 7 activation functions

there are 12 activation functions（6 + 6（5 LeakyReLU））
swish
lrelu_03 negative_slope=0.3
lrelu_02 negative_slope=0.2
gelu
signed_sqrt

'''