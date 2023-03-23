"""
Encoder: Stack of transformer encoder blocks
"""
import torch
from torch import nn

from transformer.transformer import TransformerEncoderBlock


class Encoder(nn.Module):

    def __init__(self, k, heads, depth, seq_length, num_tokens):

        super().__init__()

        self.tokenEmbeddings = nn.Embedding(num_tokens, k)
        self.positionEmbeddings = nn.Embedding(seq_length, k)

        self.transformerBlocks = []

        for i in range(depth):
            transformerBlock = TransformerEncoderBlock(k, heads)
            self.transformerBlocks.append(transformerBlock)

        self.transformerBlocks = nn.Sequential(*self.transformerBlocks)

    def forward(self, x):

        tokens = self.tokenEmbeddings(x)
        b, t, k = tokens.size()
        print(b, t, k)
        positions = torch.arange(t)
        positions = self.positionEmbeddings(positions)[None, :, :].expand(b, t, k)

        encoderInputs = tokens + positions

        encoderOutputs = self.transformerBlocks(encoderInputs)

        return encoderOutputs
