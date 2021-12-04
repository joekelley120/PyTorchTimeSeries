import torch
import torch.nn as nn
from Utility.TDL import insert_tdl
from Utility.FullyConnectedLayer import FullyConnectedLayer
import math


class NARMAXCell(torch.nn.Module):

    __constants__ = ['input_delay_size',
                     'output_delay_size',
                     'hidden_size',
                     'input_size',
                     'output_size',
                     'zero_input_delay',
                     'activation_type',
                     'layers']

    def __init__(self, input_delay_size, output_delay_size, error_delay_size, hidden_size,
                 input_size, output_size, zero_input_delay=False, activation_type='tanh',
                 layers=1):

        """
        NARMAX Cell.

        :param input_delay_size: size of input TDL
        :param output_delay_size: size of output TDL
        :param error_delay_size: size of error TDL
        :param hidden_size: number of neurons in hidden layer
        :param input_size: number of inputs
        :param output_size: number of outputs
        :param zero_input_delay: no input delay
        :param activation_type: hidden layer activation type (ie 'tanh', 'sigmoid', 'relu', and 'linear')
        :param layers: number of hidden layers (Default 1)
        """

        super(NARMAXCell, self).__init__()

        # NARX architecture
        self.input_delay_size = input_delay_size
        self.output_delay_size = output_delay_size
        self.error_delay_size = error_delay_size
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
                                                      self.output_size * self.error_delay_size +
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

    def forward(self, input, itdl, otdl, etdl, output=None):
        # type: (Tensor, Tensor, Tensor, Tensor, Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor]

        """
        Forward pass through NARMAX cell.

        Type of forward pass depends on if an output (y(t)) is provided. If output is None then
        predict the next output (y_hat(t)) and insert that output prediction into the output
        TDL (otdl) this is common for multi-step prediction. Otherwise insert the
        output (y(t)) into the output TDL (otdl) this is referred to as loading.

        :param input: input (u(t))
        :param output: output (y(t))
        :param itdl: input TDL
        :param otdl: output TDL
        :param etdl: error TDL
        :return: outputs (y_hat(t)), input TDL, output TDL, error TDL
        """

        if output is None:
            outputs, itdl, otdl, etdl = self.__pred_and_update(input, itdl, otdl, etdl)
        else:
            outputs, itdl, otdl, etdl = self.__loading(input, output, itdl, otdl, etdl)

        return outputs, itdl, otdl, etdl

    def __loading(self, input, output, itdl, otdl, etdl):
        # type: (Tensor, Tensor, Tensor, Tensor, Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor]

        """
        Predict 1-step ahead for the NARMAX cell. The measurement (y(t)) is inserted into the output
        TDL(otdl) every time this method is called. We refer to as the loading phase.
        Loading is the process of inserting inputs and outputs measurements into their corresponding
        TDL before performing a multi-step prediction.

        :param input: input (u(t))
        :param output: output (y(t))
        :param itdl: input TDL
        :param otdl: output TDL
        :param etdl: error TDL
        :return: outputs (y_hat(t)), input TDL, output TDL, error TDL
        """

        # Predict y_hat(t)
        a2 = self.__pred(input, itdl, otdl, etdl)

        # Calculate prediction error
        e = output - a2.view(-1, 1, self.output_size)

        # Insert input (u(t)) into tap-delay
        itdl = insert_tdl(itdl, input.view(-1, self.input_size, 1), shift=self.input_size, dim=1)

        # Update output (y(t)) tap-delay
        otdl = insert_tdl(otdl, output.view(-1, self.output_size, 1), shift=self.output_size, dim=1)

        # Update error (e(t)) tap-delay
        etdl = insert_tdl(etdl, e.view(-1, self.output_size, 1), shift=self.output_size, dim=1)

        return a2.view(-1, 1, self.output_size), itdl, otdl, etdl

    def __pred_and_update(self, input, itdl, otdl, etdl):
        # type: (Tensor, Tensor, Tensor, Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor]

        """
        Predict 1-step ahead and update TDL for the NARMAX cell. The prediction (y_hat(t)) is
        inserted into the output TDL (otdl) every time this method is called. This method should
        be called multiple times for multi-step prediction.

        :param input: input (u(t))
        :param itdl: input TDL
        :param otdl: output TDL
        :param etdl: error TDL
        :return: outputs (y_hat(t)), input TDL, output TDL, error TDL
        """

        # Predict y_hat(t)
        a2 = self.__pred(input, itdl, otdl, etdl)

        # Insert input (u(t)) into tap-delay
        itdl = insert_tdl(itdl, input.view(-1, self.input_size, 1), shift=self.input_size, dim=1)

        # Update output (y_hat(t)) tap-delay
        otdl = insert_tdl(otdl, a2, shift=self.output_size, dim=1)

        # Update error (e(t)) tap-delay
        e = a2 * 0.0
        etdl = insert_tdl(etdl, e, shift=self.output_size, dim=1)

        return a2.view(-1, 1, self.output_size), itdl, otdl, etdl

    def __pred(self, input, itdl, otdl, etdl):
        # type: (Tensor, Tensor, Tensor, Tensor) -> Tuple[Tensor]

        """
        Predict 1-step ahead for the NARMAX cell.

        :param input: input (u(t))
        :param itdl: input TDL
        :param otdl: output TDL
        :param etdl: error TDL
        :return: outputs (y_hat(t))
        """

        # Set the input tap-delay
        if self.zero_input_delay:
            delay = torch.cat([itdl, input.view(-1, self.input_size, 1)], dim=1)
        else:
            delay = itdl

        hidden = torch.cat([delay, otdl, etdl], dim=1)

        # Loop through all fully connected layers
        for i in range(self.layers):
            hidden = self.fully[i].forward(hidden)

        # Output Layer Calculation
        n2 = torch.matmul(self.lw, hidden) + self.b2

        # Output Layer is Linear Transformation
        a2 = n2

        return a2

    def simulate(self, input, itdl, otdl, etdl, variance):
        # type: (Tensor, Tensor, Tensor, Tensor, flaot) -> Tuple[Tensor, Tensor, Tensor, Tensor]

        """
        Simulate the NARMAX model with random errors.

        :param input: input (u(t))
        :param itdl: input TDL
        :param otdl: output TDL
        :param etdl: error TDL
        :param variance: noise variance
        :return: outputs (y_hat(t)), input TDL, output TDL, error TDL
        """

        # Predict y_hat(t)
        a2 = self.__pred(input, itdl, otdl, etdl)
        e = torch.randn_like(a2) * math.sqrt(variance)
        a2 = a2 + e

        # Insert input (u(t)) into tap-delay
        itdl = insert_tdl(itdl, input.view(-1, self.input_size, 1), shift=self.input_size, dim=1)

        # Update output (y_hat(t)) tap-delay
        otdl = insert_tdl(otdl, a2, shift=self.output_size, dim=1)

        # Update error (e(t)) tap-delay
        etdl = insert_tdl(etdl, e, shift=self.output_size, dim=1)

        return a2.view(-1, 1, self.output_size), itdl, otdl, etdl


class NARMAX(nn.Module):

    def __init__(self, input_delay_size, output_delay_size, error_delay_size, hidden_size,
                 input_size, output_size, zero_input_delay=False, activation_type='tanh',
                 layers=1):

        """
        NARMAX Model.

        :param input_delay_size: size of input TDL
        :param output_delay_size: size of output TDL
        :param error_delay_size: size of error TDL
        :param hidden_size: number of neurons in hidden layer
        :param input_size: number of inputs
        :param output_size: number of outputs
        :param zero_input_delay: no input delay
        :param activation_type: hidden layer activation type (ie 'tanh', 'sigmoid', 'relu', and 'linear')
        :param layers: number of hidden layers (Default 1)
        """

        super(NARMAX, self).__init__()

        self.input_delay_size = input_delay_size
        self.output_delay_size = output_delay_size
        self.error_delay_size = error_delay_size
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.output_size = output_size
        self.activation_type = activation_type
        self.layers = layers

        cell = NARMAXCell(input_delay_size, output_delay_size, error_delay_size, hidden_size, input_size, output_size,
                          zero_input_delay, activation_type, layers)
        self.cell = cell

    def loading(self, inputs, outputs, itdl, otdl, etdl):
        # type: (Tensor, Tensor, Tensor, Tensor, Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor]

        """
        Perform lading of inputs, outputs, and errors into the NARMAX's TDL. The loading phase runs
        from 0 to time step 't'.

        :param inputs: inputs [u(0) ..... u(t)]
        :param outputs: outputs [y(0) ..... y(t)]
        :param itdl: input TDL
        :param otdl: output TDL
        :param etdl: error TDL
        :return: output [y_hat(0) .... y_hat(t)], input TDL, output TDL, error TDL
        """

        output_tensor = torch.zeros_like(inputs)

        for input, output, time in zip(inputs.split(1, 1), outputs.split(1, 1), range(inputs.size()[1])):
            prediction, itdl, otdl, etdl = self.cell(input, itdl, otdl, etdl, output)
            output_tensor[:, time: time + 1] = prediction

        return output_tensor, itdl, otdl, etdl

    def forward(self, inputs, itdl, otdl, etdl):
        # type: (Tensor, Tensor, Tensor, Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor]

        """
        Perform predictions k-steps ahead for for NARMAX. The multi-step prediction runs from time step 't' to 't+H'
        with H being the maximum prediction horizon.

        :param inputs: inputs [u(t+1) ..... u(t+H)] (H is maximum prediction horizon)
        :param itdl: input TDL
        :param otdl: output TDL
        :param etdl: error TDL
        :return: output [y_hat(t+1) .... y_hat(t+H)] (H is maximum prediction horizon), input TDL, output TDL, error TDL
        """

        output_tensor = torch.zeros_like(inputs)

        for input, time in zip(inputs.split(1, 1), range(inputs.size()[1])):
            prediction, itdl, otdl, etdl = self.cell(input, itdl, otdl, etdl)
            output_tensor[:, time: time + 1] = prediction

        return output_tensor, itdl, otdl, etdl

    def simulate(self, inputs, itdl, otdl, etdl, variance):
        # type: (Tensor, Tensor, Tensor, Tensor, float) -> Tuple[Tensor, Tensor, Tensor, Tensor]

        """
        Simulate the NARMAX. The multi-step simulation runs from time step 't' to 't+H'
        with H being the maximum prediction horizon.

        :param inputs: inputs [u(t+1) ..... u(t+H)] (H is maximum prediction horizon)
        :param itdl: input TDL
        :param otdl: output TDL
        :param etdl: error TDL
        :param variance: variance in error
        :return: output [y_hat(t+1) .... y_hat(t+H)] (H is maximum prediction horizon), input TDL, output TDL, error TDL
        """

        output_tensor = torch.zeros_like(inputs)

        for input, time in zip(inputs.split(1, 1), range(inputs.size()[1])):
            prediction, itdl, otdl, etdl = self.cell.simulate(input, itdl, otdl, etdl, variance)
            output_tensor[:, time: time + 1] = prediction

        return output_tensor, itdl, otdl, etdl
