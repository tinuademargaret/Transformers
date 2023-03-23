import torch

from transformer.network import TransformerNetwork

# using small sized matrix as I'll be testing it on my cpu

# batch size of 2, seq_length of 4
x = torch.LongTensor([[0, 1, 2, 3], [0, 1, 2, 3]])
y = torch.LongTensor([[3, 1, 2, 0], [3, 1, 2, 0]])

# embedding size
k = 8

# encoder input
e_b, e_t = x.size()

# decoder input
d_b, d_t = x.size()

encoderDepth = 2
decoderDepth = 3
heads = 4

transformerNetwork = TransformerNetwork(
    k,
    heads,
    encoderDepth,
    decoderDepth,
    e_t,
    e_t,
    d_t,
    d_t
)

if __name__ == '__main__':
    output = transformerNetwork(x, y)
    print(output)
