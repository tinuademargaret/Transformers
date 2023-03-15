"""
Decoder: Stack of transformer decoder blocks
"""

import torch
from torch import nn

from transformer.transformer import TransformerDecoderBlock


class Decoder(nn.Module):

    def __init__(self, k, heads, depth, seq_length, num_tokens):

        super().__init__()

        self.tokenEmbeddings = nn.Embedding(num_tokens, k)
        self.positionEmbeddings = nn.Embedding(seq_length, k)

        self.transformerBlocks = []

        for i in range(depth):
            transformerBlock = TransformerDecoderBlock(k, heads)
            self.transformerBlocks.append(transformerBlock)

        self.transformerBlocks = nn.Sequential(*self.transformerBlocks)

    def forward(self, x, encoderOutputs):

        tokens = self.tokenEmbeddings(x)
        b, t, k = tokens.size()
        positions = self.positionEmbeddings(
            torch.arange(t)[None, :, :].expand(b, t, k))

        decoderInputs = tokens + positions

        decoderOutputs = self.transformerBlocks(decoderInputs, encoderOutputs)

        return decoderOutputs
