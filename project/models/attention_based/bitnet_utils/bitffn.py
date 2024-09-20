from torch import Tensor, nn

from .bitlinear import BitLinear


class BitFeedForward(nn.Module):
    def __init__(self, n_input, n_hidden, n_output, dropout=0.2, activation=nn.GELU()):
        super(BitFeedForward, self).__init__()

        self.layer = nn.Sequential(
            BitLinear(n_input, n_hidden),
            activation,
            nn.Dropout(dropout),
            BitLinear(n_hidden, n_output),
        )

    def forward(self, x: Tensor) -> Tensor:
        """
        Performs the forward pass of the BitFeedForward module.

        Args:
            x (Tensor): The input tensor.

        Returns:
            Tensor: The output tensor after applying the feed-forward transformation.

        """
        return self.layer(x)
