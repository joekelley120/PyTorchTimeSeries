import torch
import torch.nn as nn


class Dense(torch.nn.Module):

    __constants__ = ['input_size',
                     'hidden_size',
                     'activation_type']

    def __init__(self, input_size, output_size, activation_type='tanh', weight_factory=0.01):

        """
        Dense Layer.

        :param input_size: input size
        :param output_size: output size
        :param activation_type: hidden layer activation type (ie 'tanh', 'sigmoid', 'relu', and 'linear')
        """

        super(Dense, self).__init__()

        self.w = nn.Parameter(weight_factory * torch.randn(output_size, input_size,
                              requires_grad=True, dtype=torch.float64))

        self.b = nn.Parameter(weight_factory * torch.randn(output_size, 1,
                              requires_grad=True, dtype=torch.float64))

        # activation type
        if activation_type is 'tanh':
            self.activation = nn.Tanh()
        elif activation_type is 'sigmoid':
            self.activation = nn.Sigmoid()
        elif activation_type is 'relu':
            self.activation = nn.ReLU()
        else:
            self.activation = lambda x: x

    def forward(self, input):
        # type: (Tensor) -> Tuple[Tensor]

        """
        Forward pass through a Dense Layer.

        :param input: input
        """

        n = torch.matmul(self.w, input) + self.b
        a = self.activation(n)

        return a
