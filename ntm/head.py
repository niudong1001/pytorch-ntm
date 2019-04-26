"""NTM Read and Write Heads."""
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np


def _split_cols(mat, lengths):
    """Split a 2D matrix to variable length columns."""
    assert mat.size()[1] == sum(lengths), "Lengths must be summed to num columns"
    # np.cumsum是在某个轴上累加
    # 参考：https://blog.csdn.net/banana1006034246/article/details/78841461
    # print(lengths)  # [20, 1, 1, 3, 1, 20, 20]
    l = np.cumsum([0] + lengths)
    # print(l)  # [ 0 20 21 22 25 26 46 66]
    results = []
    for s, e in zip(l[:-1], l[1:]):
        results += [mat[:, s:e]]
    return results


class NTMHeadBase(nn.Module):
    """An NTM Read/Write Head."""

    def __init__(self, memory, controller_size):
        """Initilize the read/write head.

        :param memory: The :class:`NTMMemory` to be addressed by the head.
        :param controller_size: The size of the internal representation.
        """
        super(NTMHeadBase, self).__init__()

        self.memory = memory # memory对象
        self.N, self.M = memory.size()
        self.controller_size = controller_size

    def create_new_state(self, batch_size):
        raise NotImplementedError

    def register_parameters(self):
        raise NotImplementedError

    def is_read_head(self):
        return NotImplementedError

    def _address_memory(self, k, β, g, s, γ, w_prev):
        # Handle Activations
        # 相当于对于LSTM的输出做一个再处理
        k = k.clone()
        # softplus是平滑版本的relu，
        # 参考：https://blog.csdn.net/bqw18744018044/article/details/81193241
        β = F.softplus(β)
        g = F.sigmoid(g)
        # 这里直接采用了softmax，而不是论文中的另一种方法
        s = F.softmax(s, dim=1)
        # 保证sharpen值不小于1
        γ = 1 + F.softplus(γ)

        w = self.memory.address(k, β, g, s, γ, w_prev)

        return w


class NTMReadHead(NTMHeadBase):
    def __init__(self, memory, controller_size):
        super(NTMReadHead, self).__init__(memory, controller_size)

        # Corresponding to k, β, g, s, γ sizes from the paper
        # k：头产生的key向量，其会跟Memory中每条记忆进行相似性比较，并计算权重向量
        # β：头产生的key强度值常数，会在根据k相似性产生权重向量的时候，增加每一项的参与强度，Paper(8)
        # g：一个插值门常数，值范围[0,1]，根据其计算出新的权值 w_{t,g} <= g * w_{t, c} + (1-g) * w_{t-1}，其中w_{t, c}为根据k/β计算出的内容向量，w_{t-1}为上一个时刻控制器产生的权值向量
        # s: 移位权值向量，是可能移位值上的概率分布，[-1, 0, 1]
        # γ：sharpen值常数，最后权值做归一化的时候，作为指数值
        self.read_lengths = [self.M, 1, 1, 3, 1]
        # nn.Linear(in_features, out_features)
        # y=wx+b, weight=Paramter(torch.Tensor(out_feature, in_feature))
        # bias = Paramter(torch.Tensor(out_feature))
        self.fc_read = nn.Linear(controller_size, sum(self.read_lengths))
        self.reset_parameters()

    def create_new_state(self, batch_size):
        # The state holds the previous time step address weightings
        return torch.zeros(batch_size, self.N)

    def reset_parameters(self):
        # Initialize the linear layers
        nn.init.xavier_uniform_(self.fc_read.weight, gain=1.4)
        nn.init.normal_(self.fc_read.bias, std=0.01)

    def is_read_head(self):
        return True

    def forward(self, embeddings, w_prev):
        """NTMReadHead forward function.

        :param embeddings: input representation of the controller.
        :param w_prev: previous step state
        """
        o = self.fc_read(embeddings)
        # 先从LSTM输出上获得相应的每一个量
        k, β, g, s, γ = _split_cols(o, self.read_lengths)

        # Read from memory
        w = self._address_memory(k, β, g, s, γ, w_prev)
        r = self.memory.read(w)

        return r, w


class NTMWriteHead(NTMHeadBase):
    def __init__(self, memory, controller_size):
        super(NTMWriteHead, self).__init__(memory, controller_size)

        # Corresponding to k, β, g, s, γ, e, a sizes from the paper
        # e: erase向量
        # a: add向量
        self.write_lengths = [self.M, 1, 1, 3, 1, self.M, self.M]
        self.fc_write = nn.Linear(controller_size, sum(self.write_lengths))
        self.reset_parameters()

    def create_new_state(self, batch_size):
        return torch.zeros(batch_size, self.N)

    def reset_parameters(self):
        # Initialize the linear layers
        nn.init.xavier_uniform_(self.fc_write.weight, gain=1.4)
        nn.init.normal_(self.fc_write.bias, std=0.01)

    def is_read_head(self):
        return False

    def forward(self, embeddings, w_prev):
        """NTMWriteHead forward function.

        :param embeddings: input representation of the controller.
        :param w_prev: previous step state
        """
        o = self.fc_write(embeddings)
        k, β, g, s, γ, e, a = _split_cols(o, self.write_lengths)

        # e should be in [0, 1]
        e = F.sigmoid(e)  # 作用在最后一维上
        # print(e.shape)  # torch.Size([-1, 20])

        # Write to memory
        w = self._address_memory(k, β, g, s, γ, w_prev)
        self.memory.write(w, e, a)

        return w
