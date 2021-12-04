import torch
import torch.nn as nn


class GeneralDotProductAttention(torch.nn.Module):

    __constants__ = ['input_size']

    def __init__(self, input_size):

        """
        General Dot Product Attention.

        :param input_size: attention input size
        """

        super(GeneralDotProductAttention, self).__init__()

        self.input_size = input_size

        self.w = nn.Parameter(0.01 * torch.randn(self.input_size, self.input_size,
                              requires_grad=True, dtype=torch.float64))

    def forward(self, query, keys, values):
        # type: (Tensor, Tensor, Tensor) -> Tuple[Tensor]

        """
        Forward pass through a Dot Product Attention

        :param query: query
        :param keys: keys
        :param values: values
        """

        keys_t = torch.transpose(keys, 1, 2)
        a1 = torch.matmul(self.w, query)
        n2 = torch.matmul(keys_t, a1)
        a2 = torch.softmax(n2, dim=1)
        a3 = torch.matmul(values, a2)

        return a3
