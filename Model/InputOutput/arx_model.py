import torch
import torch.nn as nn
from Utility.TDL import insert_tdl


class ARXCell(torch.nn.Module):

    __constants__ = ['input_delay_size',
                     'output_delay_size',
                     'input_size',
                     'output_size',
                     'zero_input_delay']

    def __init__(self, input_delay_size, output_delay_size, input_size, output_size,
                 zero_input_delay=False):

        """
        ARX Cell.

        :param input_delay_size: size of input tap-delay
        :param output_delay_size: size of output tap-delay
        :param input_size: number of inputs
        :param output_size: number of outputs
        :param zero_input_delay: no input delay
        """

        super(ARXCell, self).__init__()

        # NARX architecture
        self.input_delay_size = input_delay_size
        self.output_delay_size = output_delay_size
        self.input_size = input_size
        self.output_size = output_size
        self.zero_input_delay = zero_input_delay

        # Input TDL Weight
        self.iw = nn.Parameter(0.01 * torch.randn(1, self.input_delay_size * self.input_size +
                               self.zero_input_delay * self.input_size,
                               requires_grad=True, dtype=torch.float64))

        # Output TDL Weight
        self.ow = nn.Parameter(0.01 * torch.randn(1, self.output_delay_size * self.output_size,
                               requires_grad=True, dtype=torch.float64))

        self.b = nn.Parameter(0.01 * torch.randn(1, 1, requires_grad=True, dtype=torch.float64))

    def forward(self, input, itdl, otdl, output=None):
        # type: (Tensor, Tensor, Tensor, Tensor) -> Tuple[Tensor, Tensor, Tensor]

        """
        Forward pass through ARX cell.

        Type of forward pass depends on if an output (y(t)) is provided. If output is None then
        predict the next output (y_hat(t)) and insert that output prediction into the output
        TDL (otdl) this is common for multi-step ahead prediction. Otherwise, insert the
        output (y(t)) into the output TDL (otdl) this is referred to as loading.

        :param input: input (u(t))
        :param output: output (y(t))
        :param itdl: input TDL
        :param otdl: output TDL
        :return: outputs (y_hat(t)), input tDL, output TDL
        """

        if output is None:
            outputs, itdl, otdl = self.__pred_and_update(input, itdl, otdl)
        else:
            outputs, itdl, otdl = self.__loading(input, output, itdl, otdl)

        return outputs, itdl, otdl

    def __loading(self, input, output, itdl, otdl):
        # type: (Tensor, Tensor, Tensor, Tensor) -> Tuple[Tensor, Tensor, Tensor]

        """
        Predict 1-step ahead for the ARX cell. The measurement (y(t)) is inserted into the output
        TDL (otdl) every time this method is called, which we refer to as the loading phase.
        Loading is the process of inserting inputs and outputs measurements into their corresponding
        TDL before performing a multi-step prediction.

        :param input: input (u(t))
        :param output: output (y(t))
        :param itdl: input TDL
        :param otdl: output TDL
        :return: outputs (y_hat(t)), input tDL, output TDL
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
        Predict 1-step ahead and update TDL for the ARX cell. The prediction (y_hat(t)) is
        inserted into the output TDL (otdl) every time this method is called. This method should
        be called multiple times for multi-step ahead prediction.

        :param input: input (u(t))
        :param itdl: input TDL
        :param otdl: output TDL
        :return: outputs (y_hat(t)), input tDL, output TDL
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

        return torch.matmul(self.iw, delay) + torch.matmul(self.ow, otdl) + self.b


class ARX(nn.Module):

    def __init__(self, input_delay_size, output_delay_size, input_size, output_size, zero_input_delay=False):

        """
        ARX Model.

        :param input_delay_size: size of input TDL
        :param output_delay_size: size of output TDL
        :param input_size: number of inputs
        :param output_size: number of outputs
        :param zero_input_delay: no input delay
        """

        super(ARX, self).__init__()

        self.input_delay_size = input_delay_size
        self.output_delay_size = output_delay_size
        self.input_size = input_size
        self.output_size = output_size

        self.cell = ARXCell(input_delay_size, output_delay_size, input_size, output_size, zero_input_delay)

    def loading(self, inputs, outputs, itdl, otdl):
        # type: (Tensor, Tensor, Tensor, Tensor) -> Tuple[Tensor, Tensor, Tensor]

        """
        Perform lading of inputs and outputs into the ARX's TDL. The loading phase runs
        from 0 to time step 't'.

        :param inputs: inputs [u(0) ..... u(t)]
        :param outputs: outputs [y(0) ..... y(t)]
        :param itdl: input TDL
        :param otdl: output TDL
        :return: output [y_hat(0) .... y_hat(t)], input tDL, output TDL
        """

        output_tensor = torch.zeros_like(inputs)

        for input, output, time in zip(inputs.split(1, 1), outputs.split(1, 1), range(inputs.size()[1])):
            prediction, itdl, otdl = self.cell(input, itdl, otdl, output)
            output_tensor[:, time: time + 1] = prediction

        return output_tensor, itdl, otdl

    def forward(self, inputs, itdl, otdl):
        # type: (Tensor, Tensor, Tensor) -> Tuple[Tensor, Tensor, Tensor]

        """
        Perform predictions k-steps ahead for for ARX. The multi-step prediction runs from time step 't' to 't+H'
        with H being the maximum prediction horizon.

        :param inputs: inputs [u(t+1) ..... u(t+H)] (H is maximum prediction horizon)
        :param itdl: input TDL
        :param otdl: output TDL
        :return: output [y_hat(t+1) .... y_hat(t+H)] (H is maximum prediction horizon), input tDL, output TDL
        """

        output_tensor = torch.zeros_like(inputs)

        for input, time in zip(inputs.split(1, 1), range(inputs.size()[1])):
            prediction, itdl, otdl = self.cell(input, itdl, otdl)
            output_tensor[:, time: time + 1] = prediction

        return output_tensor, itdl, otdl
