#!/usr/bin/env python
from __future__ import absolute_import, division, print_function

import math

import torch
from torch import nn
from torch.autograd import Function
from torch.autograd.function import once_differentiable
from torch.nn.modules.utils import _pair

import _ext as _backend


class _DCNv2(Function):
    @staticmethod
    def forward(
        ctx, input, offset, mask, weight, bias, stride, padding, dilation, deformable_groups
    ):
        ctx.stride = _pair(stride)
        ctx.padding = _pair(padding)
        ctx.dilation = _pair(dilation)
        ctx.kernel_size = _pair(weight.shape[2:4])
        ctx.deformable_groups = deformable_groups
        output = _backend.dcn_v2_forward(
            input,
            weight,
            bias,
            offset,
            mask,
            ctx.kernel_size[0],
            ctx.kernel_size[1],
            ctx.stride[0],
            ctx.stride[1],
            ctx.padding[0],
            ctx.padding[1],
            ctx.dilation[0],
            ctx.dilation[1],
            ctx.deformable_groups,
        )
        ctx.save_for_backward(input, offset, mask, weight, bias)
        return output

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        input, offset, mask, weight, bias = ctx.saved_tensors
        grad_input, grad_offset, grad_mask, grad_weight, grad_bias = _backend.dcn_v2_backward(
            input,
            weight,
            bias,
            offset,
            mask,
            grad_output,
            ctx.kernel_size[0],
            ctx.kernel_size[1],
            ctx.stride[0],
            ctx.stride[1],
            ctx.padding[0],
            ctx.padding[1],
            ctx.dilation[0],
            ctx.dilation[1],
            ctx.deformable_groups,
        )

        return grad_input, grad_offset, grad_mask, grad_weight, grad_bias, None, None, None, None

    @staticmethod
    def symbolic(
        g, input, offset, mask, weight, bias, stride, padding, dilation, deformable_groups
    ):
        from torch.nn.modules.utils import _pair

        stride = _pair(stride)
        padding = _pair(padding)
        dilation = _pair(dilation)
        # as of trt 7, the dcn operation will be translated again by modifying the onnx file
        # so the exporting code is kept to resemble the forward()
        return g.op(
            "DCNv2_2",
            input,
            offset,
            mask,
            weight,
            bias,
            stride_i=stride,
            padding_i=padding,
            dilation_i=dilation,
            deformable_groups_i=deformable_groups,
        )


dcn_v2_conv = _DCNv2.apply


class DCNv2(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride,
        padding,
        dilation=1,
        deformable_groups=1,
    ):
        super(DCNv2, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = _pair(kernel_size)
        self.stride = _pair(stride)
        self.padding = _pair(padding)
        self.dilation = _pair(dilation)
        self.deformable_groups = deformable_groups

        self.weight = nn.Parameter(torch.Tensor(out_channels, in_channels, *self.kernel_size))
        self.bias = nn.Parameter(torch.Tensor(out_channels))
        self.reset_parameters()

    def reset_parameters(self):
        self.weight.data.normal_(0, 0.02)
        self.bias.data.zero_()

    def forward(self, input, offset, mask):
        assert (
            2 * self.deformable_groups * self.kernel_size[0] * self.kernel_size[1]
            == offset.shape[1]
        )
        assert self.deformable_groups * self.kernel_size[0] * self.kernel_size[1] == mask.shape[1]
        return dcn_v2_conv(
            input,
            offset,
            mask,
            self.weight,
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.deformable_groups,
        )


class HarmonicConv(DCNv2):
    def __init__(self, in_channels, out_channels, kernel_size=7, stride=1, padding=3, dilation=1, deformable_groups=7):
        super(HarmonicConv, self).__init__(in_channels, out_channels, (7, 1), stride, (3, 0), dilation, deformable_groups)

        self.offset = None
        self.anchor = 7
        self.kernel_size = kernel_size
        self.deformable_groups = deformable_groups
        self.temporal_conv = nn.Conv2d(out_channels, out_channels, (1, 7), 1, (0, 3))
        self.mixing_conv = nn.Conv1d(deformable_groups, deformable_groups, 1, 1, 0)

    def init_offset(self, height):
        self.offset = torch.zeros((1, self.deformable_groups * self.kernel_size * 1 * 2, height, 1))
        for anchor in range(self.anchor):
            for i in range(self.kernel_size):
                for j in range(1):
                    anchor_ptr = anchor * self.kernel_size * 1 * 2
                    for h in range(height):
                        self.offset[:, anchor_ptr + (2 * i), h, :] = \
                            - i + (h+1)/(anchor+1) * (i+1) - (h+1)

    def forward(self, input):
        offset = torch.zeros((input.shape[0], self.deformable_groups * self.kernel_size * 1 * 2, input.shape[2], input.shape[3])).to(input.device)
        if self.offset is None:
            self.init_offset(input.shape[2])

        offset = offset + self.offset.to(input.device)
        mask = torch.ones((input.shape[0], 7 * 7 * 1, input.shape[2], input.shape[3])).to(input.device)
        out = dcn_v2_conv(input, offset, mask, self.weight, self.bias,
                          self.stride, self.padding, self.dilation, self.deformable_groups)
        out = self.temporal_conv(out)
        out_shape = out.shape
        out = out.reshape(input.shape[0], self.deformable_groups, -1)
        out = self.mixing_conv(out)
        out = out.reshape(*out_shape)
        return out
