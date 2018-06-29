import torch
import torch.nn as nn
import torch.nn.functional as F

from collections import namedtuple, OrderedDict, Iterable

class Conv2d_tf(nn.Conv2d):
    def __init__(self, *args, **kwargs):
        super(Conv2d_tf, self).__init__(*args, **kwargs)
        self.padding = kwargs.get('padding', 'SAME')
        kwargs['padding'] = 0
        if not isinstance(self.stride, Iterable):
            self.stride = (self.stride, self.stride)
        if not isinstance(self.dilation, Iterable):
            self.dilation = (self.dilation, self.dilation)

    def forward(self, input):
        # from https://github.com/pytorch/pytorch/issues/3867
        if self.padding == 'VALID':
            return F.conv2d(input, self.weight, self.bias, self.stride,
                            padding=0,
                            dilation=self.dilation, groups=self.groups)
        input_rows = input.size(2)
        filter_rows = self.weight.size(2)
        effective_filter_size_rows = (filter_rows - 1) * self.dilation[0] + 1
        out_rows = (input_rows + self.stride[0] - 1) // self.stride[0]
        padding_rows = max(0, (out_rows - 1) * self.stride[0] + effective_filter_size_rows -
                                input_rows)
        # padding_rows = max(0, (out_rows - 1) * self.stride[0] +
        #                         (filter_rows - 1) * self.dilation[0] + 1 - input_rows)
        rows_odd = (padding_rows % 2 != 0)
        # same for padding_cols
        input_cols = input.size(3)
        filter_cols = self.weight.size(3)
        effective_filter_size_cols = (filter_cols - 1) * self.dilation[1] + 1
        out_cols = (input_cols + self.stride[1] - 1) // self.stride[1]
        padding_cols = max(0, (out_cols - 1) * self.stride[1] + effective_filter_size_cols -
                                input_cols)
        # padding_cols = max(0, (out_cols - 1) * self.stride[1] +
        #                         (filter_cols - 1) * self.dilation[1] + 1 - input_cols)
        cols_odd = (padding_cols % 2 != 0)

        if rows_odd or cols_odd:
            input = F.pad(input, [0, int(cols_odd), 0, int(rows_odd)])

        return F.conv2d(input, self.weight, self.bias, self.stride,
                        padding=(padding_rows // 2, padding_cols // 2),
                        dilation=self.dilation, groups=self.groups)

def _make_divisible(v, divisor, min_value=None):
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v

def depth_multiplier_v2(depth,
                        multiplier,
                        divisible_by=8,
                        min_depth=8):
    d = depth
    return _make_divisible(d * multiplier, divisible_by,
                                                    min_depth)