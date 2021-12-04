import torch


def insert_tdl(data, inp, shift=1, dim=0):
    # type: (Tensor, Tensor, int, int) -> Tensor

    """
    Insert an element into a tensor and remove the last
    element.

    :param data: tensor
    :param inp: input
    :param dim: dimension
    :param shift: shift
    :return: new tensor
    """

    return torch.cat([data[:, shift:], inp], dim=dim)
