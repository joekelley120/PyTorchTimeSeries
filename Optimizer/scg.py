import torch
from torch.optim import Optimizer


class SCG(Optimizer):

    """
    Implements Scaled Conjugate Gradient algorithm, inspired by the article
    Møller, Martin F. A scaled conjugate gradient algorithm for fast supervised
    learning. Aarhus University, Computer Science Department, 1990.

    ... warning::
        This optimizer doesn't support per-parameter options and
        parameters groups.

    Arguments:
        min_grad (float): minimum gradient norm
        _lambda (float): between 0 < lambda < 1.0e-8. (default 5.0e-7)
        sigma (float): between 0 < sigma < 1.0e-6. (default 5.0e-5)

    """

    def __init__(self, params, _lambda=5.0e-7, sigma=5.0e-5, min_grad=1e-10, num_x=1000):

        """
        Initializer for implementation of Conjugate Gradient
        algorithm.

        :param _lambda: between 0 < lambda < 1.0e-8. (default 5.0e-7)
        :param sigma: between 0 < sigma < 1.0e-6. (default 5.0e-5)
        :param params: model parameters
        :param num_x: epoch number for resetting gradient
        :param min_grad: minimum gradient norm
        """

        defaults = dict(
            min_grad=min_grad,
        )

        super(SCG, self).__init__(params, defaults)

        if len(self.param_groups) != 1:
            raise ValueError("SCG doesn't support per-parameter options "
                             "(parameter group)")

        self._params = self.param_groups[0]['params']

        self.state.setdefault('success', 1)
        self.state.setdefault('lambdab', torch.tensor(0))
        self.state.setdefault('lambda', torch.tensor(_lambda))
        self.state.setdefault('lambdak', torch.tensor(0))
        self.state.setdefault('epoch', 0)
        self.state.setdefault('sigma', torch.tensor(sigma))
        self.state.setdefault('perf', torch.tensor(0))
        self.state.setdefault('deltak', torch.tensor(0))
        self.state.setdefault('nrmsqr_dx', torch.tensor(0))
        self.state.setdefault('gx', None)
        self.state.setdefault('dx', None)
        self.state.setdefault('norm_dx', None)
        self.state.setdefault('gx_old', None)
        self.state.setdefault('x', None)
        self.state.setdefault('norm_gx', 0)
        self.state.setdefault('completed',False)
        self.state.setdefault('num_x', num_x)
        self.state.setdefault('reset', False)

    def _clone_param(self):

        """
        Clone model parameters.

        :return: cloned model parameters.
        """

        return [p.clone() for p in self._params if p.requires_grad]

    @staticmethod
    def _clone_passed_params(params):

        """
        Clone passed in parameters.

        :param params: parameters
        :return: cloned parameters
        """

        return [p.clone() for p in params if p.requires_grad]

    def _set_param(self, params_data):

        """
        Set model parameters.

        :param params_data: desired parameters for model.
        :return: none
        """

        i = 0
        for p in self._params:
            if p.requires_grad:
                p.data.copy_(params_data[i].data)
                i += 1

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

        # Model parameters
        params = self._clone_param()

        # Load set training parameters
        group = self.param_groups[0]
        min_grad = group['min_grad']

        # Perform a golden section line search
        final_loss = self._scaled_conjugate_gradient(closure, params, min_grad)

        return final_loss

    def _scaled_conjugate_gradient(self, closure, params, min_grad=1e-10):

        """
        Implementation of Scaled Conjugate Gradient line search.

        :param closure: A closure evaluates the model's gradients
                        and returns the loss.
        :param min_grad: minimum gradient norm
        :return: loss
        """

        sigma = self.state['sigma']
        epoch = self.state['epoch']
        success = self.state['success']
        lambdab = self.state['lambdab']
        _lambda = self.state['lambda']
        lambdak = self.state['lambdak']
        perf = self.state['perf']
        deltak = self.state['deltak']
        nrmsqr_dx = self.state['nrmsqr_dx']
        gx = self.state['gx']
        dx = self.state['dx']
        norm_dx = self.state['norm_dx']
        gx_old = self.state['gx_old']
        x = self.state['x']
        norm_gx = self.state['norm_gx']
        num_x = self.state['num_x']
        reset = self.state['reset']

        def grad():

            """
            Calculate gradient.

            :return: gradients
            """

            # Calculate loss and gradients
            _loss = closure()

            # Create a list of gradients
            g = [p.grad for p in self._params if p.requires_grad]

            # return loss value and vector for the gradients
            return _loss.item(), torch.nn.utils.parameters_to_vector(g)

        # Initialize training parameters for first epoch
        if epoch is 0 or reset:
            # Disable reset
            reset = False

            # Convert parameters to vector
            x = torch.nn.utils.parameters_to_vector(params)

            # Calculate loss and gradient
            self._set_param_vector(x)
            perf, gx = grad()

            norm_gx = torch.dot(gx, gx).sqrt()
            gx_old = gx.clone()

            # Initialize search direction and norm
            dx = -gx.clone()
            nrmsqr_dx = torch.dot(dx, dx)
            norm_dx = nrmsqr_dx.sqrt()

            # Initialize training parameters and flag
            success = 1
            lambdab = 0
            lambdak = _lambda

        # If success is true, calculate second order information
        if success is 1:
            sigmak = sigma / norm_dx
            x_temp = x + sigmak * dx
            self._set_param_vector(x_temp)
            perf_temp, gx_temp = grad()
            sk = (gx_temp - gx) / sigmak
            deltak = torch.dot(dx, sk)

        # Scale deltak
        deltak = deltak + (lambdak - lambdab) * nrmsqr_dx

        # If deltak <= 0 then make the Hessian matrix positive
        # definite
        if deltak <= 0:
            lambdab = 2.0 * (lambdak - deltak / nrmsqr_dx)
            deltak = -deltak + lambdak * nrmsqr_dx
            lambdak = lambdab

        # Calculate step size
        muk = torch.dot(-dx, gx)
        alphak = muk / deltak

        # Calculate the comparison parameter
        x_temp = x + alphak * dx
        self._set_param_vector(x_temp)
        perf_temp, gx_temp = grad()
        difk = 2.0 * deltak * (perf - perf_temp) / muk.pow(2)

        # If difk >= 0 then a successful reduction in error can
        # be made
        if difk >= 0:
            gx_old = gx.clone()
            x = x_temp.clone()
            perf = perf_temp
            gx = gx_temp.clone()
            norm_gx = torch.dot(gx, gx).sqrt()
            lambdab = 0
            success = 1

            # Restart the algorithm every num_X iterations
            if epoch % num_x is 0:
                dx = -gx.clone()
            else:
                betak = (torch.dot(gx, gx) - torch.dot(gx, gx_old)) / muk
                dx = -gx + betak * dx

            nrmsqr_dx = torch.dot(dx, dx)
            norm_dx = nrmsqr_dx.sqrt()

            if difk >= 0.75:
                lambdak *= 0.25

        else:
            self._set_param_vector(x)
            lambdab = lambdak
            success = 0

        if difk < 0.25:
            lambdak = lambdak + deltak * (1.0 - difk) / nrmsqr_dx

        if lambdak >= 10000:
            reset = True

        if norm_gx < min_grad:
            self.state['completed'] = True

        self.state['epoch'] = epoch + 1
        self.state['success'] = success
        self.state['lambdab'] = lambdab
        self.state['lambda'] = _lambda
        self.state['lambdak'] = lambdak
        self.state['perf'] = perf
        self.state['deltak'] = deltak
        self.state['nrmsqr_dx'] = nrmsqr_dx
        self.state['gx'] = gx
        self.state['dx'] = dx
        self.state['norm_dx'] = norm_dx
        self.state['gx_old'] = gx_old
        self.state['x'] = x
        self.state['norm_gx'] = norm_gx
        self.state['reset'] = reset

        return perf
