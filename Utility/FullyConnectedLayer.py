import torch
import torch.nn as nn


class FullyConnectedLayer(torch.nn.Module):

    __constants__ = ['input_size',
                     'hidden_size',
                     'activation_type']

    def __init__(self, input_size, hidden_size, activation_type='tanh'):

        """
        Fully Connected Layer.

        :param input_size: number of inputs
        :param hidden_size: number of neurons in hidden layer
        :param activation_type: hidden layer activation type (ie 'tanh', 'sigmoid', 'relu', and 'linear')
        """

        super(FullyConnectedLayer, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size

        self.w = nn.Parameter(0.01 * torch.randn(self.hidden_size, self.input_size,
                              requires_grad=True, dtype=torch.float64))

        self.b = nn.Parameter(0.01 * torch.randn(self.hidden_size, 1,
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
        Forward pass through a Fully Connected Layer.

        :param input: input
        """

        n = torch.matmul(self.w, input) + self.b
        a = self.activation(n)

        return a
