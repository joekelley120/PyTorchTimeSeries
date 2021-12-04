import torch


class DotProductAttention(torch.nn.Module):

    def __init__(self):

        """
        Dot Product Attention.
        """

        super(DotProductAttention, self).__init__()

    def forward(self, query, keys, values):
        # type: (Tensor, Tensor, Tensor) -> Tuple[Tensor]

        """
        Forward pass through a Dot Product Attention

        :param query: query
        :param keys: keys
        :param values: values
        """

        keys_t = torch.transpose(keys, 1, 2)
        n1 = torch.matmul(keys_t, query)
        a1 = torch.softmax(n1, dim=1)
        a2 = torch.matmul(values, a1)

        return a2
