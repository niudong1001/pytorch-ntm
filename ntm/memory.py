"""An NTM's memory implementation."""
import torch
import torch.nn.functional as F
from torch import nn
import numpy as np
import sys


def _convolve(w, s):
    """Circular convolution implementation."""
    assert s.size(0) == 3
    # torch.cat(https://blog.csdn.net/qq_39709535/article/details/80803003)
    # print(w.shape, s.shape)  # torch.Size([N]) torch.Size([3])
    t = torch.cat([w[-1:], w, w[:1]])  # 在w前后拼上合适的值，padding便于卷积操作
    # F.conv1d（http://www.ituring.com.cn/article/468202）
    c = F.conv1d(t.view(1, 1, -1), s.view(1, 1, -1)).view(-1)
    return c


class NTMMemory(nn.Module):
    """Memory bank for NTM."""
    def __init__(self, N, M):
        """Initialize the NTM Memory matrix.

        The memory's dimensions are (batch_size x N x M).
        Each batch has it's own memory matrix.

        :param N: Number of rows in the memory.
        :param M: Number of columns/features in the memory.
        """
        super(NTMMemory, self).__init__()

        self.N = N
        self.M = M

        # The memory bias allows the heads to learn how to initially address
        # memory locations by content
        # pytorch将参数保存成OrderedDict，其包含两种nn.Parameter和buffer中的参数，
        # 后者不会被更新，register_buffer则是用来创建buffer参数
        self.register_buffer('mem_bias', torch.Tensor(N, M))

        # Initialize memory bias
        # 为什么使用这种初始化方式？
        stdev = 1 / (np.sqrt(N + M))
        nn.init.uniform_(self.mem_bias, -stdev, stdev)

    def reset(self, batch_size):
        """Initialize memory from bias, for start-of-sequence."""
        self.batch_size = batch_size
        # 注意记忆矩阵也存在多对
        self.memory = self.mem_bias.clone().repeat(batch_size, 1, 1)

    def size(self):
        return self.N, self.M

    def read(self, w):
        """Read from memory (according to section 3.1)."""
        # unsqueeze在某个维度上增加单维度
        # squeeze去掉某个单维
        # 参考：https://blog.csdn.net/flysky_jay/article/details/81607289
        # 下面先在w的1(N)维度上展开一维，做了乘法后再去掉第1维的结果
        return torch.matmul(w.unsqueeze(1), self.memory).squeeze(1)

    def write(self, w, e, a):
        """write to memory (according to section 3.2)."""
        self.prev_mem = self.memory
        self.memory = torch.Tensor(self.batch_size, self.N, self.M)
        erase = torch.matmul(w.unsqueeze(-1), e.unsqueeze(1))
        add = torch.matmul(w.unsqueeze(-1), a.unsqueeze(1))
        self.memory = self.prev_mem * (1 - erase) + add

    def address(self, k, β, g, s, γ, w_prev):
        """NTM Addressing (according to section 3.3).

        Returns a softmax weighting over the rows of the memory matrix.

        :param k: The key vector.
        :param β: The key strength (focus).
        :param g: Scalar interpolation gate (with previous weighting).
        :param s: Shift weighting.
        :param γ: Sharpen weighting scalar.
        :param w_prev: The weighting produced in the previous time step.
        """
        # Content focus
        wc = self._similarity(k, β)

        # Location focus
        wg = self._interpolate(w_prev, wc, g)
        ŵ = self._shift(wg, s)
        w = self._sharpen(ŵ, γ)

        return w

    def _similarity(self, k, β):
        # print(k, k.shape)  # (-1, M)
        # print(β, β.shape)  # (-1, 1)
        k = k.view(self.batch_size, 1, -1)
        # print(self.memory.shape)  # torch.Size([1, 128, 20])
        # TODO: Maybe change to another sim function
        w = F.softmax(β * F.cosine_similarity(self.memory + 1e-16, k + 1e-16, dim=-1), dim=1)
        return w

    def _interpolate(self, w_prev, wc, g):
        return g * wc + (1 - g) * w_prev

    def _shift(self, wg, s):
        # print(wg.shape)  # torch.Size([-1, N])
        # print(s.shape, s) # torch.Size([-1, 3]) tensor([[0.3113, 0.3994, 0.2894]])
        result = torch.zeros(wg.size())
        for b in range(self.batch_size):
            # 每个batch上进行卷积操作
            result[b] = _convolve(wg[b], s[b])
        return result

    def _sharpen(self, ŵ, γ):
        w = ŵ ** γ
        # torch.div除法
        w = torch.div(w, torch.sum(w, dim=1).view(-1, 1) + 1e-16)
        return w
