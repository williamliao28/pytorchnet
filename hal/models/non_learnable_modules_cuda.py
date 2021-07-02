import math
import time # test elapsed time
import torch
import torch.nn as nn
import torch.nn.functional as F

import nl_module_cuda

_TRACK_RUNNING_STATS = False

__all__ = ['NonLearnableLayer']


# modified from pytorchnet's implementation
# https://github.com/human-analysis/pytorchnet/blob/master/models/lbcresnet.py
def LBConv(in_planes, out_planes, kernel_size=3, stride=1,
           padding=1, dilation=1, groups=1, bias=False, sparsity=0.1, binary=True):
    conv2d = nn.Conv2d(
        in_planes, out_planes, kernel_size=kernel_size,
        stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias,
    )
    conv2d.weight.requires_grad = False
    if conv2d.bias:
        conv2d.bias.requires_grad = False

    if binary:
        conv2d.weight.fill_(0.0)
        num = conv2d.weight.numel()
        shape = conv2d.weight.size()
        index = torch.Tensor(math.floor(sparsity * num)).random_(num).int()
        conv2d.weight.resize_(num)
        for i in range(index.numel()):
            conv2d.weight[index[i]] = torch.bernoulli(torch.Tensor([0.5])) * 2 - 1
        conv2d.weight.resize_(shape)

    return conv2d


# non-learnable layer
class NonLearnableLayer(nn.Module):
    def __init__(self, C_in, C_out, kernel_size, stride, padding, groups, sparsity=0.5, binary=True, drop=0.0):
        super(NonLearnableLayer, self).__init__()
        self.conv = LBConv(in_planes=C_in, out_planes=C_in, kernel_size=kernel_size, stride=stride,
                           padding=padding, groups=groups, bias=False, sparsity=sparsity, binary=binary)
        self.maxpool = nn.MaxPool2d(kernel_size=kernel_size, stride=stride, padding=padding)
        self.avgpool = nn.AvgPool2d(kernel_size=kernel_size, stride=stride, padding=padding)
        self.linear_combination = nn.Conv2d(3 * C_in, C_out, kernel_size=1, padding=0, bias=False)
        self.drop = drop

    def forward(self, x):
        poolsize = torch.tensor([float(self.maxpool.kernel_size),float(self.maxpool.kernel_size)])
        stride   = torch.tensor([float(self.maxpool.stride),float(self.maxpool.stride)])
        padding  = torch.tensor([float(self.maxpool.padding),float(self.maxpool.padding)])
        #input("Press Enter to continue...")
        t0 = time.process_time()
        x_conv = self.conv(x)
        #t1 = time.process_time() - t0
        print(f"Time elapsed for local binary convolution: {time.process_time() - t0:.16f} seconds")
        #d_x1,d_x2,d_x3 = nl_module_cuda.nl_forward(x,self.conv(x),poolsize,stride,padding)
        #print(f"d_x1 size: {d_x1.shape}")
        #print(f"d_x2 size: {d_x2.shape}")
        #print(f"d_x3 size: {d_x3.shape}")
        #input("Press Enter to continue...")
        #x1 = F.relu(self.conv(x), inplace=True)
        t0 = time.process_time()
        x1 = F.relu(x_conv, inplace=True)
        x2 = self.maxpool(x)
        x3 = self.avgpool(x)
        #print(f"x1 size: {x1.shape}")
        #print(f"x2 size: {x2.shape}")
        #print(f"x3 size: {x3.shape}")
        #print(x1[0][0])
        #print(f"Check x1: {torch.all(torch.abs(x1-d_x1)<1e-8)}")
        #if not torch.all(torch.abs(x1-d_x1)<1e-8) :
        #    input("Press Enter to continue...")
        #print(x2[0][0])
        #print(f"Check x2: {torch.all(torch.abs(x2-d_x2)<1e-8)}")
        #if not torch.all(torch.abs(x2-d_x2)<1e-8) :
        #    input("Press Enter to continue...")
        #print(x3[0][0])
        #print(f"Check x3: {torch.all(torch.abs(x3-d_x3)<1e-8)}")
        #if not torch.all(torch.abs(x3-d_x3)<1e-8) :
        #    input("Press Enter to continue...")
        #input("Press Enter to continue...")
        x_cat = torch.cat([x1, x2, x3], dim=1)
        #t1 = time.process_time() - t0
        print(f"Time elapsed for pytorch: {time.process_time() - t0:.16f} seconds")
        t0 = time.process_time()
        [d_x_cat] = nl_module_cuda.nl_forward_withcat(x,x_conv,poolsize,stride,padding)
        #t1 = time.process_time() - t0
        print(f"Time elapsed for cuda kernel: {time.process_time() - t0:.16f} seconds")
        input("Press Enter to continue...")
        print(f"Check size of x_cat, d_x_cat, conv(x), x1, x2, x3:")
        print(f"x size: {x.shape}")
        print(f"x_cat size: {x_cat.shape}")
        print(f"d_x_cat size: {d_x_cat.shape}")
        print(f"conv(x) size: {x_conv.shape}")
        print(f"x1 size: {x1.shape}")
        print(f"x2 size: {x2.shape}")
        print(f"x3 size: {x3.shape}")
        input("Press Enter to continue...")
        if not torch.all(torch.abs(x_cat-d_x_cat)<1e-8) :
            print(f"kernel_size = {poolsize}")
            print(f"stride = {stride}")
            print(f"padding = {padding}")
            print(f"d_x_cat size: {d_x_cat.shape}")
            print(f"x_cat size: {x_cat.shape}")
            print(f"Check x_cat: {torch.all(torch.abs(torch.lt(torch.sub(x_cat,d_x_cat),1e-8)))}")
            print(x_cat[0][0])
            print(d_x_cat[0][0])
            input("Press Enter to continue...")
        x_cat = d_x_cat
        if self.drop > 0:
            x_cat = F.dropout(x_cat, p=self.drop, training=self.training)
        return self.linear_combination(x_cat)


# internal non-learnable layer
class _NonLearnableLayer(nn.Module):
    def __init__(self, C_in, kernel_size, stride, padding, groups, sparsity=0.5, binary=True):
        super(_NonLearnableLayer, self).__init__()
        self.conv = LBConv(in_planes=C_in, out_planes=C_in, kernel_size=kernel_size, stride=stride,
                           padding=padding, groups=groups, bias=False, sparsity=sparsity, binary=binary)
        self.maxpool = nn.MaxPool2d(kernel_size=kernel_size, stride=stride, padding=padding)
        self.avgpool = nn.AvgPool2d(kernel_size=kernel_size, stride=stride, padding=padding)

    def forward(self, x):
        x1 = F.relu(self.conv(x), inplace=True)
        x2 = self.maxpool(x)
        x3 = self.avgpool(x)
        return torch.cat([x1, x2, x3], dim=1)
