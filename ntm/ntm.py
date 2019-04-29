#!/usr/bin/env python
import torch
from torch import nn
import torch.nn.functional as F


class NTM(nn.Module):
    """A Neural Turing Machine."""
    def __init__(self, num_inputs, num_outputs, controller, memory, heads):
        """Initialize the NTM.

        :param num_inputs: External input size.
        :param num_outputs: External output size.
        :param controller: :class:`LSTMController`
        :param memory: :class:`NTMMemory`
        :param heads: list of :class:`NTMReadHead` or :class:`NTMWriteHead`

        Note: This design allows the flexibility of using any number of read and
              write heads independently, also, the order by which the heads are
              called in controlled by the user (order in list)
        """
        super(NTM, self).__init__()

        # Save arguments
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.controller = controller
        self.memory = memory
        self.heads = heads

        self.N, self.M = memory.size()
        # num_inputs, num_outputs
        _, self.controller_size = controller.size()

        # Initialize the initial previous read values to random biases
        # 下面初始化的是每个读头能读取的值
        self.num_read_heads = 0
        self.init_r = []
        for head in heads:
            if head.is_read_head():
                init_r_bias = torch.randn(1, self.M) * 0.01
                self.register_buffer("read{}_bias".format(self.num_read_heads), init_r_bias.data)
                self.init_r += [init_r_bias]
                self.num_read_heads += 1

        assert self.num_read_heads > 0, "heads list must contain at least a single read head"

        # Initialize a fully connected layer to produce the actual output:
        # 这里是产生输出的前向网络
        #   [controller_output; previous_reads ] -> output
        self.fc = nn.Linear(self.controller_size + self.num_read_heads * self.M, num_outputs)
        self.reset_parameters()

    def create_new_state(self, batch_size):
        # 每次初始化值都是一样
        # 这里初始化的是读头的读取初始值（每个读头维度为(M,)），不可训练！
        init_r = [r.clone().repeat(batch_size, 1) for r in self.init_r]
        # 这里初始化的是控制器(LSTM)的c_0与h_0状态，可训练！
        controller_state = self.controller.create_new_state(batch_size)
        # 这里初始化的是每个读/写头的初始化权值向量，不可训练！
        heads_state = [head.create_new_state(batch_size) for head in self.heads]

        return init_r, controller_state, heads_state

    def reset_parameters(self):
        # Initialize the linear layer
        nn.init.xavier_uniform_(self.fc.weight, gain=1)
        nn.init.normal_(self.fc.bias, std=0.01)

    def forward(self, x, prev_state):
        """NTM forward function.

        :param x: input vector (batch_size x num_inputs)
        :param prev_state: The previous state of the NTM
        """
        # Unpack the previous state
        prev_reads, prev_controller_state, prev_heads_states = prev_state

        # Use the controller to get an embeddings
        # 注意这里的输入是外部输入+读头读到的数据
        inp = torch.cat([x] + prev_reads, dim=1)
        controller_outp, controller_state = self.controller(inp, prev_controller_state)
        # torch.Size([1, 100]) 2
        # print(controller_outp.shape, len(controller_state))
        # torch.Size([1, 1, 100]) torch.Size([1, 1, 100])
        # print(controller_state[0].shape, controller_state[1].shape)

        # Read/Write from the list of heads
        reads = []
        heads_states = []
        for head, prev_head_state in zip(self.heads, prev_heads_states):
            # 按照顺序运行读/写头
            if head.is_read_head():
                r, head_state = head(controller_outp, prev_head_state)
                reads += [r]
            else:
                head_state = head(controller_outp, prev_head_state)
            heads_states += [head_state]

        # Generate Output
        inp2 = torch.cat([controller_outp] + reads, dim=1)
        o = F.sigmoid(self.fc(inp2))
        # print(o.shape)  # torch.Size([1, 8])

        # Pack the current state
        state = (reads, controller_state, heads_states)

        return o, state
