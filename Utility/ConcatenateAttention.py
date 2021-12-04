import torch
import torch.nn as nn


class ConcatenateAttention(torch.nn.Module):

    __constants__ = ['input_size',
                     'hidden_size']

    def __init__(self, input_size, hidden_size):

        """
        Concatenate Attention.

        :param input_size: attention input size
        :param hidden_size: number of hidden neurons in concatenation attention
        """

        super(ConcatenateAttention, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size

        self.w42 = nn.Parameter(0.01 * torch.randn(self.hidden_size, 2 * self.input_size,
                                requires_grad=True, dtype=torch.float64))

        self.b4 = nn.Parameter(0.01 * torch.randn(self.hidden_size, 1,
                               requires_grad=True, dtype=torch.float64))

        self.w54 = nn.Parameter(0.01 * torch.randn(1, self.hidden_size,
                                requires_grad=True, dtype=torch.float64))

        self.b5 = nn.Parameter(0.01 * torch.randn(1, 1, requires_grad=True,
                               dtype=torch.float64))

    def forward(self, query, keys, values):
        # type: (Tensor, Tensor, Tensor) -> Tuple[Tensor]

        """
        Forward pass through a Dot Product Attention

        :param query: query
        :param keys: keys
        :param values: values
        """

        query_repeat = query.repeat(1, 1, keys.size(2))
        cat_query_key = torch.cat([keys, query_repeat], dim=1)
        n4 = torch.matmul(self.w42, cat_query_key) + self.b4
        a4 = torch.tanh(n4)
        n5 = torch.matmul(self.w54, a4) + self.b5
        n5 = torch.transpose(n5, 1, 2)
        a5 = torch.softmax(n5, dim=1)
        a6 = torch.matmul(values, a5)

        return a6
