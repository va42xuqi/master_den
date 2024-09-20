import torch
import torch.nn as nn


class MaskedMSELoss(nn.Module):

    def __init__(self):
        super(MaskedMSELoss, self).__init__()

        self.mse_loss = nn.MSELoss(reduction="none")

    def _sequence_mask(self, shape, valid_lens):
        device = valid_lens.device
        mask = (
            torch.ones((shape), device=device)
            * torch.arange(shape[1], device=device)[None, :, None]
        )
        mask = mask < valid_lens[:, None, None]
        return mask

    def forward(self, src, tgt, valid_lens):
        valid_lens = torch.as_tensor(valid_lens)
        loss = self.mse_loss(src, tgt)
        mask = self._sequence_mask(src.shape, valid_lens)
        loss = loss.masked_fill(mask == 0, 0)
        loss = loss.sum() / mask.sum()
        return loss


if __name__ == "__main__":

    loss_fn = MaskedMSELoss()

    X = torch.rand((4, 33, 2))
    Y = torch.rand((4, 33, 2))

    predict = 10

    valid_lens = range(X.shape[0])
    loss = loss_fn(X, Y, valid_lens)
    print(loss)
