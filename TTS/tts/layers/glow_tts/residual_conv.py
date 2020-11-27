# Adapted from: https://github.com/janvainer/speedyspeech/blob/master/code/layers.py

import torch
from torch import nn
from torch.nn import functional as F

class Mish(nn.Module):
    """ Implementation for Mish activation Function: https://www.bmvc2020-conference.com/assets/papers/0928.pdf"""
    def __init__(self):
        super().__init__()
    def forward(self, x):
        return x * torch.tanh(F.softplus(x))

class Conv1d(nn.Conv1d):
    """A wrapper around Conv1d, that works on (batch, time, channels)"""

    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, dilation=1, groups=1, bias=True, padding=0):
        super(Conv1d, self).__init__(in_channels=in_channels, out_channels=out_channels,
                                     kernel_size=kernel_size, stride=stride, dilation=dilation,
                                     groups=groups, bias=bias, padding=padding)

    def forward(self, x):
        return super().forward(x.transpose(2,1)).transpose(2,1)

class ZeroTemporalPad(nn.ZeroPad2d):
    """Pad sequences to equal lentgh in the temporal dimension"""
    def __init__(self, kernel_size, dilation, causal=False):
        total_pad = (dilation * (kernel_size - 1))
        begin = total_pad // 2
        end = total_pad - begin
        super(ZeroTemporalPad, self).__init__((0, 0, begin, end))

class FreqNorm(nn.BatchNorm1d):
    """Normalize separately each frequency channel in spectrogram and batch,
    Compare to layer norm:
        Layer_norm: (t - t.mean(-1, keepdim=True))/torch.sqrt(t.var(-1, unbiased=False, keepdim=True)+1e-05)
        -> layer norm normalizes across all frequencies for each timestep independently of batch
        => LayerNorm: Normalize each freq. bin wrt to other freq bins in the same timestep -> time independent, batch independent, freq deendent
        => FreqNorm: Normalize each freq. bin wrt to the same freq bin across time and batch -> time dependent, other freq independent
    """
    def __init__(self, in_out_channels, affine=True, track_running_stats=True, momentum=0.1):
        super(FreqNorm, self).__init__(in_out_channels, affine=affine, track_running_stats=track_running_stats, momentum=momentum)

    def forward(self, x):
        return super().forward(x.transpose(2,1)).transpose(2,1)

class ConvResidualBlock(nn.Module):
    """Implements Conv-> Mish -> FreqNorm"""
    def __init__(self, in_out_channels, kernel_size, dilation, n=2):
        super(ConvResidualBlock, self).__init__()
        self.rblocks = [
            nn.Sequential(
                Conv1d(in_out_channels, in_out_channels, kernel_size, dilation=dilation),
                ZeroTemporalPad(kernel_size, dilation),
                Mish(),
                FreqNorm(in_out_channels),  # Normalize after activation.
            )
            for i in range(n)
        ]
        self.rblocks = nn.Sequential(*self.rblocks)

    def forward(self, x):
        x = x + self.rblocks(x)
        return x

class ConvResidualBlocks(nn.Module):
    """Residual convolutional block as in https://arxiv.org/pdf/2008.03802.pdf
    Improviments:
    We changed the activation function from ReLU to Mish.
    
    Args:
        in_out_channels (int): number of input/output channels.
        kernel_size (int): convolution kernel size.
        dilations (list): Residual Block dilation, its control number block too
    """
    def __init__(self, in_out_channels=128, kernel_size=4, dilations=4*[1,2,4]+[1], apply_mask=False):
        super().__init__()

        self.apply_mask = apply_mask

        self.prenet = nn.Sequential(Conv1d(in_out_channels, in_out_channels, 1), Mish())

        self.residual_blocks = nn.Sequential(*[
            ConvResidualBlock(in_out_channels, kernel_size, dilation)
            for dilation in dilations])

        self.conv1 = Conv1d(in_out_channels, in_out_channels, 1)
        self.conv_norm = nn.Sequential(
            Mish(),
            FreqNorm(in_out_channels),
            Conv1d(in_out_channels, in_out_channels, 1))

    def forward(self, x, x_mask=None):
        x = x.transpose(1,2)

        res = self.prenet(x)
        res = self._apply_mask(res, x_mask)

        x = self.residual_blocks(x)
        x = self._apply_mask(x, x_mask)

        x = self.conv1(x) + res
        x = self.conv_norm(x)
        x = self._apply_mask(x, x_mask)

        return x.transpose(1,2)

    def _apply_mask(self, x, x_mask):
        if self.apply_mask:
            x = x.transpose(1,2) * x_mask
            x = x.transpose(1,2)
        return x

