import torch
from torch.optim import Optimizer


class PARTICLE(Optimizer):

    """
    Implements Gradient based Particle Optimization.

    .. warning::
        This optimizer doesn't support per-parameter options and
        parameters groups.

    Arguments:
        ds (float) - step size for particle swarm

    """

    def __init__(self, params, ds=0.01, contraction=0.8, expansion=1.2, number_batches=1):

        """
        Initializer for implementation of particle optimization.

        :param params: model parameters
        :param ds: step size for particle swarm
        :param contraction: coefficient for contraction of swarm
        :param expansion: coefficient for expansion of swarm
        :param number_batches: number of batches during training
        """

        defaults = dict(
            ds=ds,
            contraction=contraction,
            expansion=expansion,
            number_batches=number_batches,
        )

        super(PARTICLE, self).__init__(params, defaults)

        if len(self.param_groups) != 1:
            raise ValueError("Particle doesn't support per-parameter options "
                             "(parameter group)")

        self._params = self.param_groups[0]['params']

        self.state.setdefault('ds', ds)
        self.state.setdefault('contraction', contraction)
        self.state.setdefault('expansion', expansion)
        self.state.setdefault('iteration', 0)
        self.state.setdefault('ds_min', 1e-7)
        self.state.setdefault('ds_max', 5)
        self.state.setdefault('number_batches', number_batches)
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

        :param closure: A closure evaluates the model gradient and returns a loss.
        :return: loss
        """

        assert len(self.param_groups) == 1

        # Model parameters
        params = self._clone_param()

        # Perform a step of particle
        final_loss = self._particle(closure, params)

        return final_loss

    def _particle(self, closure, params):

        """
        Perform a step of the particle optimization algorithm.

        :param closure: forward pass though network
        :return: loss
        """

        contraction = self.state['contraction']
        expansion = self.state['expansion']
        xb = self.state['xb']
        fb = self.state['fb']
        ds = self.state['ds']
        ds_min = self.state['ds_min']
        ds_max = self.state['ds_max']
        n = self.state['number_batches']
        iteration = self.state['iteration']
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

        # Convert parameters to vector
        xo = torch.nn.utils.parameters_to_vector(params)

        if iteration is 0 or reset:
            xb = xo.clone()
            fb, _ = grad()
            reset = False

        # Evaluate fitness
        self._set_param_vector(xo)
        _, gradient = grad()

        # Calculate learning rate
        step = ds / torch.sqrt(torch.sum(torch.pow(gradient, 2)))

        # Update Particle
        random = xo.new(xo.size(0)).uniform_()
        x = xo + 2.0 * random * (xb - xo) - step * gradient

        # Evaluate fitness
        self._set_param_vector(x)
        loss, gradient = grad()

        # Determine if player has improved
        if loss >= fb:
            ds = max([contraction * ds, ds_min])
            if ds is ds_min:
                reset = True
        elif loss <= fb:
            ds = min([expansion * ds, ds_max])
            fb = fb + (1 / n) * (loss - fb)
            xb = xb + (1 / n) * (x - xb)

        self._set_param_vector(x)

        self.state['contraction'] = contraction
        self.state['expansion'] = expansion
        self.state['xb'] = xb
        self.state['fb'] = fb
        self.state['ds'] = ds
        self.state['iteration'] = iteration + 1
        self.state['reset'] = reset

        return loss
