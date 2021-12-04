import torch
import torch.nn as nn
from Utility.TDL import insert_tdl
from Utility.FullyConnectedLayer import FullyConnectedLayer
import math


class NARXCell(torch.nn.Module):

    __constants__ = ['input_delay_size',
                     'output_delay_size',
                     'hidden_size',
                     'input_size',
                     'output_size',
                     'zero_input_delay',
                     'activation_type',
                     'layers']

    def __init__(self, input_delay_size, output_delay_size, hidden_size, input_size, output_size,
                 zero_input_delay=False, activation_type='tanh', layers=1):

        """
        NARX Cell.

        :param input_delay_size: size of input tap-delay
        :param output_delay_size: size of output tap-delay
        :param hidden_size: number of neurons in hidden layer
        :param input_size: number of inputs
        :param output_size: number of outputs
        :param zero_input_delay: no input delay
        :param activation_type: hidden layer activation type (ie 'tanh', 'sigmoid', 'relu', and 'linear')
        :param layers: number of hidden layers (Default 1)
        """

        super(NARXCell, self).__init__()

        # NARX architecture
        self.input_delay_size = input_delay_size
        self.output_delay_size = output_delay_size
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.output_size = output_size
        self.zero_input_delay = zero_input_delay
        self.activation_type = activation_type
        self.layers = layers

        self.add_input_delay = 0 if self.zero_input_delay is False else 1

        # Initialize fully connected layers
        self.fully = nn.ModuleList()
        for i in range(layers):
            if i is 0:
                self.fully.append(FullyConnectedLayer(self.input_size * self.input_delay_size + 
                                                      self.output_size * self.output_delay_size + 
                                                      self.add_input_delay,
                                                      self.hidden_size, self.activation_type))
            else:
                self.fully.append(FullyConnectedLayer(self.hidden_size, self.hidden_size, self.activation_type))

        # Second Layer
        self.lw = nn.Parameter(0.01 * torch.randn(self.output_size, self.hidden_size,
                               requires_grad=True, dtype=torch.float64))

        # Bias for the Second Layer
        self.b2 = nn.Parameter(0.01 * torch.randn(self.output_size, 1,
                               requires_grad=True, dtype=torch.float64))

    def forward(self, input, itdl, otdl, output=None):
        # type: (Tensor, Tensor, Tensor, Tensor) -> Tuple[Tensor, Tensor, Tensor]

        """
        Forward pass through NARX cell.

        Type of forward pass depends on if an output (y(t)) is provided. If output is None then
        predict the next output (y_hat(t)) and insert that output prediction into the output
        TDL (otdl) this is common for multi-step prediction. Otherwise insert the
        output (y(t)) into the output TDL (otdl) this is referred to as loading.

        :param input: input (u(t))
        :param output: output (y(t))
        :param itdl: input TDL
        :param otdl: output TDL
        :return: outputs (y_hat(t)), input TDL, output TDL
        """

        # Initialize fully connected layers
        if output is None:
            outputs, itdl, otdl = self.__pred_and_update(input, itdl, otdl)
        else:
            outputs, itdl, otdl = self.__loading(input, output, itdl, otdl)

        return outputs, itdl, otdl

    def __loading(self, input, output, itdl, otdl):
        # type: (Tensor, Tensor, Tensor, Tensor) -> Tuple[Tensor, Tensor, Tensor]

        """
        Predict 1-step ahead for the NARX cell. The measurement (y(t)) is inserted into the output
        TDL (otdl) every time this method is called, which we refer to as the loading phase.
        Loading is the process of inserting inputs and outputs measurements into their corresponding
        TDL before performing a multi-step prediction.

        :param input: input (u(t))
        :param output: output (y(t))
        :param itdl: input TDL
        :param otdl: output TDL
        :return: outputs (y_hat(t)), input TDL, output TDL
        """

        # Predict y_hat(t)
        a2 = self.__pred(input, itdl, otdl)

        # Insert input (u(t)) into tap-delay
        itdl = insert_tdl(itdl, input.view(-1, self.input_size, 1), shift=self.input_size, dim=1)

        # Update output (y(t)) tap-delay
        otdl = insert_tdl(otdl, output.view(-1, self.output_size, 1), shift=self.output_size, dim=1)

        return a2.view(-1, 1, self.output_size), itdl, otdl

    def __pred_and_update(self, input, itdl, otdl):
        # type: (Tensor, Tensor, Tensor) -> Tuple[Tensor, Tensor, Tensor]

        """
        Predict 1-step ahead and update TDL for the NARX cell. The prediction (y_hat(t)) is
        inserted into the output TDL (otdl) every time this method is called. This method should
        be called multiple times for multi-step prediction.

        :param input: input (u(t))
        :param itdl: input TDL
        :param otdl: output TDL
        :return: outputs (y_hat(t)), input TDL, output TDL
        """

        # Predict y_hat(t)
        a2 = self.__pred(input, itdl, otdl)

        # Insert input (u(t)) into tap-delay
        itdl = insert_tdl(itdl, input.view(-1, self.input_size, 1), shift=self.input_size, dim=1)

        # Update output (y_hat(t)) tap-delay
        otdl = insert_tdl(otdl, a2, shift=self.output_size, dim=1)

        return a2.view(-1, 1, self.output_size), itdl, otdl

    def __pred(self, input, itdl, otdl):
        # type: (Tensor, Tensor, Tensor) -> Tuple[Tensor]

        """
        Predict 1-step ahead for the NARX cell.

        :param input: input (u(t))
        :param itdl: input TDL
        :param otdl: output TDL
        :return: outputs (y_hat(t))
        """

        # Set the input tap-delay
        if self.zero_input_delay:
            delay = torch.cat([itdl, input.view(-1, self.input_size, 1)], dim=1)
        else:
            delay = itdl

        hidden = torch.cat([delay, otdl], dim=1)

        # Loop through all fully connected layers
        for i in range(self.layers):
            hidden = self.fully[i].forward(hidden)

        # Output Layer Calculation
        n2 = torch.matmul(self.lw, hidden) + self.b2

        # Output Layer is Linear Transformation
        a2 = n2

        return a2

    def simulate(self, input, itdl, otdl, variance):
        # type: (Tensor, Tensor, Tensor, flaot) -> Tuple[Tensor, Tensor, Tensor]

        """
        Simulate the NARMAX model with random errors.

        :param input: input (u(t))
        :param itdl: input TDL
        :param otdl: output TDL
        :param variance: noise variance
        :return: outputs (y_hat(t)), input TDL, output TDL
        """

        # Predict y_hat(t)
        a2 = self.__pred(input, itdl, otdl)

        # Add noise
        a2 = torch.randn_like(a2) * math.sqrt(variance) + a2

        # Insert input (u(t)) into tap-delay
        itdl = insert_tdl(itdl, input.view(-1, self.input_size, 1), shift=self.input_size, dim=1)

        # Update output (y_hat(t)) tap-delay
        otdl = insert_tdl(otdl, a2, shift=self.output_size, dim=1)

        return a2.view(-1, 1, self.output_size), itdl, otdl


class NARX(nn.Module):

    def __init__(self, input_delay_size, output_delay_size, hidden_size, input_size, output_size,
                 zero_input_delay=False, activation_type='tanh', layers=1):

        """
        NARX Model.

        :param input_delay_size: size of input TDL
        :param output_delay_size: size of output TDL
        :param hidden_size: number of neurons in hidden layer
        :param input_size: number of inputs
        :param output_size: number of outputs
        :param zero_input_delay: no input delay
        :param activation_type: hidden layer activation type (ie 'tanh', 'sigmoid', 'relu', and 'linear')
        :param layers: number of layers (Default 1)
        """

        super(NARX, self).__init__()

        self.input_delay_size = input_delay_size
        self.output_delay_size = output_delay_size
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.output_size = output_size
        self.layers = layers

        cell = NARXCell(input_delay_size, output_delay_size, hidden_size, input_size, output_size,
                        zero_input_delay, activation_type, layers)
        self.cell = cell

    def loading(self, inputs, outputs, itdl, otdl):
        # type: (Tensor, Tensor, Tensor, Tensor) -> Tuple[Tensor, Tensor, Tensor]

        """
        Perform loading of inputs and outputs into the NARX's TDL. The loading phase runs
        from 0 to time step 't'.

        :param inputs: inputs [u(0) ..... u(t)]
        :param outputs: outputs [y(0) ..... y(t)]
        :param itdl: input TDL
        :param otdl: output TDL
        :return: output [y_hat(0) .... y_hat(t)], input TDL, output TDL
        """

        output_tensor = torch.zeros_like(inputs)

        for input, output, time in zip(inputs.split(1, 1), outputs.split(1, 1), range(inputs.size()[1])):
            prediction, itdl, otdl = self.cell(input, itdl, otdl, output)
            output_tensor[:, time: time + 1] = prediction

        return output_tensor, itdl, otdl

    def forward(self, inputs, itdl, otdl):
        # type: (Tensor, Tensor, Tensor) -> Tuple[Tensor, Tensor, Tensor]

        """
        Perform predictions k-steps ahead for for NARX. The multi-step prediction runs from time step 't' to 't+H'
        with H being the maximum prediction horizon.

        :param inputs: inputs [u(t+1) ..... u(t+H)] (H is maximum prediction horizon)
        :param itdl: input TDL
        :param otdl: output TDL
        :return: output [y_hat(t+1) .... y_hat(t+H)] (H is maximum prediction horizon), input TDL, output TDL
        """

        output_tensor = torch.zeros_like(inputs)

        for input, time in zip(inputs.split(1, 1), range(inputs.size()[1])):
            prediction, itdl, otdl = self.cell(input, itdl, otdl)
            output_tensor[:, time: time + 1] = prediction

        return output_tensor, itdl, otdl

    def simulate(self, inputs, itdl, otdl, variance):
        # type: (Tensor, Tensor, Tensor, float) -> Tuple[Tensor, Tensor, Tensor]

        """
        Simulate the NARX. The multi-step simulation runs from time step 't' to 't+H'
        with H being the maximum prediction horizon.

        :param inputs: inputs [u(t+1) ..... u(t+H)] (H is maximum prediction horizon)
        :param itdl: input TDL
        :param otdl: output TDL
        :param variance: variance in error
        :return: output [y_hat(t+1) .... y_hat(t+H)] (H is maximum prediction horizon), input TDL, output TDL
        """

        output_tensor = torch.zeros_like(inputs)

        for input, time in zip(inputs.split(1, 1), range(inputs.size()[1])):
            prediction, itdl, otdl = self.cell.simulate(input, itdl, otdl, variance)
            output_tensor[:, time: time + 1] = prediction

        return output_tensor, itdl, otdl
