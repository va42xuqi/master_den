
import torch
import torch.nn.functional as F
from torch import nn


class BitLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super(BitLinear, self).__init__()

        self.in_features = in_features
        self.out_features = out_features

        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.randn(out_features))
        else:
            self.register_parameter("bias", None)

        # Learnable parameters for quantization and dequantization
        self.gamma = nn.Parameter(torch.ones(in_features))
        self.beta = nn.Parameter(torch.ones(out_features))

    def forward(self, input):
        # Apply Layer Normalization
        input_norm = F.layer_norm(input, (self.in_features,))

        # Absmax Quantization
        quant_scale = torch.max(torch.abs(input_norm), dim=1, keepdim=True).values
        input_quant = torch.sign(input_norm) * (quant_scale / self.gamma)

        # 1-bit Weights Quantization
        weight_quant = torch.sign(self.weight)

        # MatMul with 1-bit weights using torch.matmul for explicit operation
        output = torch.matmul(input_quant, weight_quant.t())

        # Adding bias if it exists
        if self.bias is not None:
            output += self.bias.unsqueeze(0).expand_as(output)

        # Dequantization with learnable parameters
        output = output * self.beta.unsqueeze(0).expand_as(output)

        return output