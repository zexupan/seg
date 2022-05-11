import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from apex import amp
import copy

EPS = 1e-8

def _clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

class avSync(nn.Module):
    def __init__(self, N=256):
        super(avSync, self).__init__()
        self.audioNet = audioNet()
        self.visualNet = visualNet()
        self.avNet = avNet()

    def forward(self, a, v):
        a = self.audioNet(a)
        v = self.visualNet(v)
        out = self.avNet(a,v)
        return out


class tcn(nn.Module):
    def __init__(self, B = 256, H = 512, P = 3, X=4):
        super(tcn, self).__init__()
        blocks = []
        for x in range(X):
            dilation = 2**x
            padding = (P - 1) * dilation // 2
            blocks += [TemporalBlock(B, H, P, stride=1,
                                     padding=padding,
                                     dilation=dilation)]
        self.net = nn.Sequential(*blocks)

    def forward(self, x):
        out = self.net(x)
        return out

class audioNet(nn.Module):
    def __init__(self, L=80, N=256):
        super(audioNet, self).__init__()
        self.conv1d_U_0 = nn.Conv1d(1, N, kernel_size=L, stride=L // 2, padding = 40, bias=False)

        self.norm_0 = ChannelWiseLayerNorm(N)

        self.tcn_1 = tcn()
        self.pool_1 = nn.AvgPool1d(4)
        self.tcn_2 = tcn()
        self.pool_2 = nn.AvgPool1d(4)
        
    def forward(self, a):
        a = torch.unsqueeze(a, 1)
        a = F.relu(self.conv1d_U_0(a))
        a = self.norm_0(a)

        a = self.tcn_1(a)
        a = self.pool_1(a)
        a = self.tcn_2(a)
        a = self.pool_2(a)
        return a

class visualNet(nn.Module):
    def __init__(self):
        super(visualNet, self).__init__()
        stacks = []
        for x in range(5):
            stacks +=[VisualConv1D(V=256, H=512)]
        self.visual_conv = nn.Sequential(*stacks)
        self.v_us = nn.Conv1d(30, 256, 1, bias=False)

    def forward(self, visual):
        visual = visual.transpose(1,2)
        visual = self.v_us(visual)
        visual = self.visual_conv(visual)
        return visual

class VisualConv1D(nn.Module):
    def __init__(self, V=256, H=512):
        super(VisualConv1D, self).__init__()
        relu_0 = nn.ReLU()
        norm_0 = GlobalLayerNorm(V)
        conv1x1 = nn.Conv1d(V, H, 1, bias=False)
        relu = nn.ReLU()
        norm_1 = GlobalLayerNorm(H)
        dsconv = nn.Conv1d(H, H, 3, stride=1, padding=1,dilation=1, groups=H, bias=False)
        prelu = nn.PReLU()
        norm_2 = GlobalLayerNorm(H)
        pw_conv = nn.Conv1d(H, V, 1, bias=False)
        self.net = nn.Sequential(relu_0, norm_0, conv1x1, relu, norm_1 ,dsconv, prelu, norm_2, pw_conv)

    def forward(self, x):
        out = self.net(x)
        return out + x

class avNet(nn.Module):
    def __init__(self, N=256):
        super(avNet, self).__init__()
        self.avconv_1 = nn.Conv1d(N*2, N, kernel_size = 1)
        self.tcn_1 = tcn(X=2)
        self.pool_1=nn.AvgPool1d(3)

        self.avconv_2 = avConv()
        self.pool_2=nn.AvgPool1d(3)

        self.avconv_3 = avConv()

        self.adpavgPool=nn.AdaptiveAvgPool1d(1)
        self.linear = nn.Linear(N, 1)

    def forward(self, a, v):
        a=F.pad(a,(0,v.size()[-1] - a.size()[-1]))
        av = torch.cat((a,v), axis = 1)
        av = self.avconv_1(av)
        av = self.tcn_1(av)
        av = self.pool_1(av)

        av = self.avconv_2(av)
        av = self.pool_2(av)
        av = self.avconv_3(av)
        av = self.adpavgPool(av).squeeze(2)
        av = self.linear(av).squeeze(1)
        return av


class avConv(nn.Module):
    def __init__(self, N = 256, dim=512):
        super(avConv, self).__init__()
        relu_0 = nn.ReLU()
        norm_0 = GlobalLayerNorm(N)
        conv1x1 = nn.Conv1d(N, dim, 1, bias=False)
        relu = nn.ReLU()
        norm_1 = GlobalLayerNorm(dim)
        dsconv = nn.Conv1d(dim, dim, 3, stride=1, dilation=1, padding = 1, groups=dim, bias=False)
        prelu = nn.PReLU()
        norm_2 = GlobalLayerNorm(dim)
        pw_conv = nn.Conv1d(dim, N, 1, bias=False)

        self.net = nn.Sequential(relu_0, norm_0, conv1x1,relu, norm_1 ,dsconv, prelu, norm_2, pw_conv)

    def forward(self, x):
        out = self.net(x)
        return out


class TemporalBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size,
                 stride, padding, dilation):
        super(TemporalBlock, self).__init__()
        conv1x1 = nn.Conv1d(in_channels, out_channels, 1, bias=False)
        prelu = nn.PReLU()
        norm = GlobalLayerNorm(out_channels)
        dsconv = DepthwiseSeparableConv(out_channels, in_channels, kernel_size,
                                        stride, padding, dilation)
        # Put together
        self.net = nn.Sequential(conv1x1, prelu, norm, dsconv)

    def forward(self, x):

        residual = x
        out = self.net(x)
        return out + residual  # look like w/o F.relu is better than w/ F.relu


class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size,
                 stride, padding, dilation):
        super(DepthwiseSeparableConv, self).__init__()
        depthwise_conv = nn.Conv1d(in_channels, in_channels, kernel_size,
                                   stride=stride, padding=padding,
                                   dilation=dilation, groups=in_channels,
                                   bias=False)

        prelu = nn.PReLU()
        norm = GlobalLayerNorm(in_channels)
        pointwise_conv = nn.Conv1d(in_channels, out_channels, 1, bias=False)
        self.net = nn.Sequential(depthwise_conv, prelu, norm,
                                 pointwise_conv)

    def forward(self, x):
        return self.net(x)

class ChannelWiseLayerNorm(nn.LayerNorm):
    @amp.float_function
    def __init__(self, *args, **kwargs):
        super(ChannelWiseLayerNorm, self).__init__(*args, **kwargs)

    @amp.float_function
    def forward(self, x):
        if x.dim() != 3:
            raise RuntimeError("{} accept 3D tensor as input".format(
                self.__name__))
        # N x C x T => N x T x C
        x = torch.transpose(x, 1, 2)
        # LN
        x = super().forward(x)
        # N x C x T => N x T x C
        x = torch.transpose(x, 1, 2)
        return x


class GlobalLayerNorm(nn.Module):
    """Global Layer Normalization (gLN)"""
    @amp.float_function
    def __init__(self, channel_size):
        super(GlobalLayerNorm, self).__init__()
        self.gamma = nn.Parameter(torch.Tensor(1, channel_size, 1))  # [1, N, 1]
        self.beta = nn.Parameter(torch.Tensor(1, channel_size,1 ))  # [1, N, 1]
        self.reset_parameters()

    @amp.float_function
    def reset_parameters(self):
        self.gamma.data.fill_(1)
        self.beta.data.zero_()

    @amp.float_function
    def forward(self, y):
        """
        Args:
            y: [M, N, K], M is batch size, N is channel size, K is length
        Returns:
            gLN_y: [M, N, K]
        """
        # TODO: in torch 1.0, torch.mean() support dim list
        mean = y.mean(dim=1, keepdim=True).mean(dim=2, keepdim=True) #[M, 1, 1]
        var = (torch.pow(y-mean, 2)).mean(dim=1, keepdim=True).mean(dim=2, keepdim=True)
        gLN_y = self.gamma * (y - mean) / torch.pow(var + EPS, 0.5) + self.beta
        return gLN_y
