import torch
from torch import nn
import torch.nn.functional as F

"""
Basic self attention

x = torch.tensor([])  # (b, t, k)

W = torch.bmm(x, x.transpose(1, 2))  # 1 -> t 2 -> k

W = F.softmax(W, dim=2)  # i.e applied across dimension 2

y = torch.bmm(W, x)
"""

# Multi head attention


class MultiHeadAttention(nn.Module):

    def __init__(self, k, heads=4, mask=False):
        super().__init__()  # Initializes the base class
        assert k % heads == 0
        self.k = k
        self.heads = heads
        self.to_queries = nn.Linear(k, k, bias=False)
        self.to_keys = nn.Linear(k, k, bias=False)
        self.to_values = nn.Linear(k, k, bias=False)
        self.unify_heads = nn.Linear(k, k)

    def forward(self, x, y=None):

        b, t, k = x.size()
        h = self.heads

        # create q, k, v of size k x k
        queries = self.to_queries(x)
        if y:
            keys = self.to_keys(y)
            values = self.to_values(y)
        else:
            keys = self.to_keys(x)
            values = self.to_values(x)

        # Split weights q, k, v to chunks of size s for h heads
        s = k // h  # size of each chunk, where a chunk belongs to a head
        queries = queries.view(b, t, h, s)
        keys = keys.view(b, t, h, s)
        values = values.view(b, t, h, s)

        # Fold h into b so that we can use torch.bmm
        queries = queries.transpose(1, 2).contigous().view(b * h, t, s)
        keys = keys.transpose(1, 2).contigous().view(b * h, t, s)
        values = values.transpose(1, 2).contigous().view(b * h, t, s)

        W = torch.bmm(queries, keys.transpose(1, 2))

        # scale the dot product
        W = W / k**(1/2)

        # Normalize with softmax
        W = F.softmax(W, dim=2)

        y = torch.bmm(W, values).view(b, h, t, s)

        # convert back to b, t, k

        y = y.transpose(1, 2).contiguous().view(b, t, s*h)

        return self.unify_heads(y)
