"""
A transformer block usually consists of
1. attention layer (decoder has 2)
2. layer normalization after the attention and
fully connected layers
3. fully connected layer
"""

from torch import nn

from transformer.attention import MultiHeadAttention


class TransformerEncoderBlock(nn.Module):

    def __init__(self, k, heads):

        super().__init__()

        self.attentionLayer = MultiHeadAttention(k, heads)

        self.normLayer1 = nn.LayerNorm(k)
        self.normLayer2 = nn.LayerNorm(k)

        self.fullyConnectedLayer = nn.Sequential(
            nn.Linear(k, 4*k),
            nn.ReLU(),
            nn.Linear(4*k, k)
        )

    def forward(self, x):

        attendedX = self.attentionLayer(x)

        x = self.normLayer1(attendedX + x)

        fedForwardX = self.fullyConnectedLayer(x)

        x = self.normLayer2(fedForwardX + x)

        return x


class TransformerDecoderBlock(nn.Module):

    def __init__(self, k, heads):

        super().__init__()

        self.attentionLayer1 = MultiHeadAttention(k, heads)
        self.attentionLayer2 = MultiHeadAttention(k, heads)

        self.normLayer1 = nn.LayerNorm(k)
        self.normLayer2 = nn.LayerNorm(k)
        self.normLayer3 = nn.LayerNorm(k)

        self.fullyConnectedLayer = nn.Sequential(
            nn.Linear(k, 4*k),
            nn.ReLU(),
            nn.Linear(4*k, k)
        )

    def forward(self, xDecoder, yEncoder):

        attendedX1 = self.attentionLayer1(xDecoder)

        x = self.normLayer1(attendedX1 + xDecoder)

        attendedX2 = self.attentionLayer2(x, yEncoder)

        x = self.normLayer2(attendedX2 + x)

        fedForwardX = self.fullyConnectedLayer(x)

        x = self.normLayer3(fedForwardX + x)

        return x


