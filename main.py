import torch

from transformer.network import TransformerNetwork
x = torch.rand(8, 32, 512)
y = torch.rand(8, 32, 512)

e_b, e_t, e_k = x.size()
d_b, d_t, d_k = x.size()
encoderDepth = 2
decoderDepth = 1
heads = 4

transformerNetwork = TransformerNetwork(
    e_k,
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


