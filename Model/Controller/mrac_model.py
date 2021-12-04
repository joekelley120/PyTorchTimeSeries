import torch
import torch.nn as nn
from Utility.TDL import insert_tdl


class ControllerConfiguration:

    def __init__(self,
                 controller_input_delay,
                 controller_reference_delay,
                 controller_output_delay,
                 controller_hidden_size,
                 plant_input_delay,
                 plant_output_delay,
                 plant_hidden_size,
                 plant_iw,
                 plant_ow,
                 plant_b1,
                 plant_lw,
                 plant_b2):

        self.controller_input_delay = controller_input_delay
        self.controller_reference_delay = controller_reference_delay
        self.controller_output_delay = controller_output_delay
        self.controller_hidden_size = controller_hidden_size
        self.plant_input_delay = plant_input_delay
        self.plant_output_delay = plant_output_delay
        self.plant_hidden_size = plant_hidden_size
        self.plant_iw = plant_iw
        self.plant_ow = plant_ow
        self.plant_b1 = plant_b1
        self.plant_lw = plant_lw
        self.plant_b2 = plant_b2


class ControllerCell(nn.Module):

    __constants__ = ['controller_input_delay',
                     'controller_reference_delay',
                     'controller_output_delay'
                     'controller_hidden_size',
                     'plant_input_delay',
                     'plant_output_delay',
                     'plant_output_delay',
                     'plant_hidden_size']

    def __init__(self, configuration):

        """
        Controller for MRAC.

        :param configuration: controller configuration
        :return: none
        """

        super(ControllerCell, self).__init__()

        self.controller_input_delay = configuration.controller_input_delay
        self.controller_reference_delay = configuration.controller_reference_delay
        self.controller_output_delay = configuration.controller_output_delay
        self.controller_hidden_size = configuration.controller_hidden_size
        self.plant_input_delay = configuration.plant_input_delay
        self.plant_output_delay = configuration.plant_output_delay
        self.plant_hidden_size = configuration.plant_hidden_size

        # CONTROLLER
        # ---------------------------------------------------------------------------
        # Input TDL Weight
        self.ciw = nn.Parameter(0.01 * torch.randn(self.controller_hidden_size, self.controller_input_delay,
                                requires_grad=True, dtype=torch.float64))

        # Reference TDL Weight
        self.crw = nn.Parameter(0.01 * torch.randn(self.controller_hidden_size, self.controller_reference_delay,
                                requires_grad=True, dtype=torch.float64))

        # Output TDL Weight
        self.cow = nn.Parameter(0.01 * torch.randn(self.controller_hidden_size, self.controller_output_delay,
                                requires_grad=True, dtype=torch.float64))

        # Bias for the First Layer of Controller
        self.cb1 = nn.Parameter(0.01 * torch.randn(self.controller_hidden_size, 1,
                                requires_grad=True, dtype=torch.float64))

        # Second Layer Controller Weight
        self.clw = nn.Parameter(0.01 * torch.randn(1, self.controller_hidden_size,
                                requires_grad=True, dtype=torch.float64))

        # Bias for the Second Layer of Controller
        self.cb2 = nn.Parameter(0.01 * torch.randn(1, 1, requires_grad=True, dtype=torch.float64))

        # PLANT
        # ---------------------------------------------------------------------------
        # Input TDL Weight
        self.piw = configuration.plant_iw.detach().data

        # Output TDL Weight
        self.pow = configuration.plant_ow.detach().data

        # Bias for the First Layer of Plant
        self.pb1 = configuration.plant_b1.detach().data

        # Second Layer Plant Weight
        self.plw = configuration.plant_lw.detach().data

        # Bias for the Second Layer of Plant
        self.pb2 = configuration.plant_b2.detach().data

    def forward(self, reference, citdl, crtdl, cotdl, pitdl, potdl):
        # type: (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]

        """
        Forward pass through MRAC.

        :param reference: reference (y(t))
        :param citdl: controller input TDL
        :param crtdl: controller reference TDL
        :param cotdl: controller output TDL
        :param pitdl: plant input TDL
        :param potdl: plant output TDL
        :return: predicted output (y(t))
        """

        # CONTROLLER
        # ---------------------------------------------------------------------------------------

        # First Layer Calculation for Controller
        n1 = (torch.matmul(self.ciw, citdl) + torch.matmul(self.crw, crtdl) +
              torch.matmul(self.cow, cotdl) + self.cb1)

        # First Layer Nonlinear Transformation
        a1 = torch.tanh(n1)

        # Second Layer Calculation for Controller
        n2 = torch.matmul(self.clw, a1) + self.cb2

        # Second Layer is Linear Transformation
        a2 = n2

        # PLANT
        # ---------------------------------------------------------------------------------------
        # Input TDL with
        pdelay = torch.cat([pitdl, a2], dim=1)

        # First Layer Calculation for Plant
        n3 = (torch.matmul(self.piw, pdelay) + torch.matmul(self.pow, potdl) + self.pb1)

        # First Layer Nonlinear Transformation
        a3 = torch.tanh(n3)

        # Second Layer Calculation for Plant
        n4 = torch.matmul(self.plw, a3) + self.pb2

        # Second Layer is Linear Transformation
        a4 = n4

        # CONTROLLER FEEDBACK
        # ---------------------------------------------------------------------------------------
        # TDL Update for Input
        citdl = insert_tdl(citdl, a2, dim=1)

        # TDL Update for Reference
        crtdl = insert_tdl(crtdl, reference, dim=1)

        # TDL Update for Output
        cotdl = insert_tdl(cotdl, a4, dim=1)

        # PLANT FEEDBACK
        # ---------------------------------------------------------------------------------------
        # TDL Update for Input
        pitdl = insert_tdl(pitdl, a2, dim=1)

        # TDL Update for Output
        potdl = insert_tdl(potdl, a4, dim=1)

        return a4, citdl, crtdl, cotdl, pitdl, potdl


class Controller(nn.Module):

    def __init__(self, configuration):

        """
        Controller for MRAC.

        :param configuration: controller configuration
        :return: none
        """

        super(Controller, self).__init__()

        self.configuration = configuration

        cell = ControllerCell(configuration)
        self.cell = cell

    def forward(self, reference, citdl, crtdl, cotdl, pitdl, potdl):
        # type: (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]

        """
        Perform predictions 'k' steps ahead for controller and plant model.

        :param reference: reference [y(t) ..... y(0)]
        :param citdl: controller input TDL
        :param crtdl: controller reference TDL
        :param cotdl: controller output TDL
        :param pitdl: plant input TDL
        :param potdl: plant output TDL
        :return:output [y(t) .... y(0)]
        """

        output_tensor = torch.zeros_like(reference)

        for ref, time in zip(reference.split(1, 1), range(reference.size()[1])):
            output, citdl, crtdl, cotdl, pitdl, potdl = self.cell(ref, citdl, crtdl, cotdl, pitdl, potdl)
            output_tensor[:, time: time + 1] = output

        return output_tensor, citdl, crtdl, cotdl, pitdl, potdl
