"""
The full network
"""

from torch import nn

from transformer.encoder import Encoder
from transformer.decoder import Decoder


class TransformerNetwork(nn.Module):

    def __init__(self,
                 k,
                 heads,
                 encoderDepth,
                 decoderDepth,
                 encoderSeqLength,
                 encoderNumTokens,
                 decoderSeqLength,
                 decoderNumTokens
                 ):

        super().__init__()
        self.encoder = Encoder(
            k,
            heads,
            encoderDepth,
            encoderSeqLength,
            encoderNumTokens
        )

        self.decoder = Decoder(
            k,
            heads,
            decoderDepth,
            decoderSeqLength,
            decoderNumTokens
        )

    def forward(self, x, y):

        encoderOutput = self.encoder(x)
        decoderOutput = self.decoder(y, encoderOutput)

        return decoderOutput
