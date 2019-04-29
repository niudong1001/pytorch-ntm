"""All in one NTM. Encapsulation of all components."""
import torch
from torch import nn
from .ntm import NTM
from .controller import LSTMController
from .head import NTMReadHead, NTMWriteHead
from .memory import NTMMemory


class EncapsulatedNTM(nn.Module):

    def __init__(self, num_inputs, num_outputs,
                 controller_size, controller_layers, num_heads, N, M):
        """Initialize an EncapsulatedNTM.

        :param num_inputs: External number of inputs.
        :param num_outputs: External number of outputs.
        :param controller_size: The size of the internal representation.
        :param controller_layers: Controller number of layers.
        :param num_heads: Number of heads.
        :param N: Number of rows in the memory bank.
        :param M: Number of cols/features in the memory bank.
        """
        super(EncapsulatedNTM, self).__init__()

        # Save args
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.controller_size = controller_size
        self.controller_layers = controller_layers
        self.num_heads = num_heads
        self.N = N
        self.M = M

        # Create the NTM components
        memory = NTMMemory(N, M)
        # 输入：外部输入 + 包含各个读头的信息
        # 输出：输出为LSTM的隐藏状态
        controller = LSTMController(num_inputs + M*num_heads, controller_size, controller_layers)
        # ModuleList类似Python中的list类型，它只是将一系列层装入列表
        heads = nn.ModuleList([])
        for i in range(num_heads):  # 读写头个数是相等的
            heads += [
                NTMReadHead(memory, controller_size),
                NTMWriteHead(memory, controller_size)
            ]

        self.ntm = NTM(num_inputs, num_outputs, controller, memory, heads)
        self.memory = memory

    def init_sequence(self, batch_size):
        """Initializing the state."""
        self.batch_size = batch_size
        # 先初始化记忆矩阵的值，注意每个batch的记忆矩阵的值都会被随机初始化
        self.memory.reset(batch_size)
        # 初始化ntm除了memory外各个部件的初始值
        self.previous_state = self.ntm.create_new_state(batch_size)

    def forward(self, x=None):
        if x is None:
            x = torch.zeros(self.batch_size, self.num_inputs)

        o, self.previous_state = self.ntm(x, self.previous_state)
        return o, self.previous_state

    def calculate_num_params(self):
        """Returns the total number of parameters."""
        num_params = 0
        for p in self.parameters():
            num_params += p.data.view(-1).size(0)
        return num_params
