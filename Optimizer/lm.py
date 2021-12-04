import torch
from torch.optim import Optimizer
import gc


class LM(Optimizer):

    """
    Hagan, Martin T., and Mohammad B. Menhaj. "Training feedforward networks
    with the Marquardt algorithm." IEEE transactions on Neural Networks 5.6
    (1994): 989-993..

    .. warning::
        This optimizer doesn't support per-parameter options and
        parameters groups.

    Arguments:
        mu: mu

    """

    def __init__(self, params, mu=0.005):

        """
        Initializer for implementation of Levenberg-Marquardt optimization
        algorithm.

        :param mu: mu
        :return none
        """

        defaults = dict()

        super(LM, self).__init__(params, defaults)

        if len(self.param_groups) != 1:
            raise ValueError("SCG doesn't support per-parameter options "
                             "(parameter group)")

        self._params = self.param_groups[0]['params']
        self._print_memory_usage = False
        self.state.setdefault('mu', mu)
        self.state.setdefault('fi', 5)

    def _clone_param(self):

        """
        Clone model parameters.

        :return: cloned model parameters.
        """

        return [p.clone() for p in self._params]

    @staticmethod
    def _clone_passed_params(params):

        """
        Clone passed in parameters.

        :param params: parameters
        :return: cloned parameters
        """

        return [p.clone() for p in params]

    def _clone_param_vector(self):

        """
        Clone model parameters.

        :return: cloned model parameters.
        """

        clone_p = [p.clone() for p in self._params]
        return torch.nn.utils.parameters_to_vector(clone_p)

    def _set_param(self, params_data):

        """
        Set model parameters.

        :param params_data: desired parameters for model.
        :return: none
        """

        for p, p_data in zip(self._params, params_data):
            p.data.copy_(p_data)

    def _set_param_vector(self, vector):

        """
        Set model parameters from a vector of parameters.

        :param vector: parameters
        :return: none
        """

        new_param = self._clone_passed_params(self._params)
        torch.nn.utils.vector_to_parameters(vector, new_param)
        self._set_param(new_param)

    def step(self, closure):

        """
        Performs a single optimization step.

        :param closure: A closure evaluates the model's gradients
                        and returns the loss.
        :return: loss
        """

        assert len(self.param_groups) == 1

        mu = self.state['mu']
        fi = self.state['fi']

        # Perform a step of Levenberg-Marquardt
        loss, mu = self._levenberg_marquardt(closure, mu, fi)

        # Set training states
        self.state['mu'] = mu

        return loss

    def _jacobian(self, closure):

        """
        Calculate Jacobian for model.

        :param closure: error function
        :return: jacobian matrix
        """

        jacobian = None
        v = None
        flag = False

        # Calculate errors
        e = closure()

        # Calculate the jacobian matrix for the prediction errors with
        # respect to model parameters
        for i in range(e.size()[0]):

            for j in range(e.size()[1]):

                # Error for loop
                error = e[i, j, :]

                # Zero the gradients
                self.zero_grad()

                # Calculate gradient for a prediction error with respect to model parameters
                # (a row in the jacobian matrix)
                grad = torch.autograd.grad(error, self._params, create_graph=True, retain_graph=True)

                gradient = None
                for k in range(len(grad)):
                    s_grad = grad[k].view(1, -1)

                    if k is 0:
                        gradient = s_grad
                    else:
                        gradient = torch.cat((gradient, s_grad), dim=1)

                # Reshape error to be matrix
                error = error.view(-1, 1)

                # Add the gradient row in the jacobian matrix
                if flag is False:
                    jacobian = gradient
                    v = error
                    flag = True
                else:
                    jacobian = torch.cat((jacobian, gradient), dim=0)
                    v = torch.cat((v, error), dim=0)

        return jacobian, v

    @staticmethod
    def _error(closure):

        """
        Calculate Error for model.

        :param closure: error function
        :return: error matrix
        """

        v = None
        flag = False

        # Calculate errors
        e = closure()

        # Reshape the error matrix
        for i in range(e.size()[0]):

            for j in range(e.size()[1]):

                # Error for loop
                error = e[i, j, :]

                # Reshape error to be matrix
                error = error.view(-1, 1)

                # Add the gradient row in the jacobian matrix
                if flag is False:
                    v = error
                    flag = True
                else:
                    v = torch.cat((v, error), dim=0)

        return v

    def _levenberg_marquardt(self, closure, mu, fi=5, device=None):

        """
        Levenberg Marquardt optimization algorithm.

        :param closure: error function
        :param mu: learning rate (adaptive)
        :param fi: learning rate increase or decrease factor
        :param device: device (cpu or gpu)
        :return: mean square error (MSE), mu for one iteration
        """

        # Calculate Jacobian
        jacobian, v = self._jacobian(closure)

        # Calculate MSE
        sum_v_old = v.pow(2).sum()

        # Setup device
        eye = torch.eye(jacobian.size()[1], dtype=torch.float64)
        if device is not None:
            eye = eye.to(device)

        # Remember old parameters
        parameters_old = self._clone_param_vector()

        while True:

            # Perform one step of Levenberg Marquardt (LM)
            step_lm = (torch.mm(torch.transpose(jacobian, 0, 1), jacobian) + mu * eye)

            inv_step_lm = torch.mm(torch.inverse(step_lm), torch.transpose(jacobian, 0, 1))
            delta_x = torch.mm(inv_step_lm, v).view(-1)

            # Update old parameters using the delta_x calculation
            parameters_new = parameters_old - delta_x

            # Set model's parameters
            self._set_param_vector(parameters_new)

            # Calculate prediction errors using the updated parameters
            v_new = self._error(closure)

            # Calculate MSE
            sum_v_new = v_new.pow(2).sum()

            # Check if new MSE is less than previous MSE.
            if sum_v_new <= sum_v_old:
                # If the MSE decreases then reduce learning rate.
                # Then break while loop since iteration of LM was a success.
                mu /= fi
                break
            else:
                # If the MSE increases then increase learning rate.
                # Then perform another setup of LM.
                mu *= fi

        return sum_v_new.item() / (v.size()[0] * v.size()[1]), mu
