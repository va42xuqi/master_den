import torch


class QuantileLoss:
    def __init__(self):
        pass

    def quantile_loss_single(self, y, predictions, quantile):
        errors = y - predictions
        quantile_loss = quantile * torch.nn.functional.relu(errors) + (
            1 - quantile
        ) * torch.nn.functional.relu(-errors)
        return quantile_loss.mean()

    def quantile_loss(self, y, predictions_list, quantiles):
        total_loss = 0
        for quantile_idx, quantile_predictions in enumerate(predictions_list):
            for quantile in quantiles:
                quantile_loss = self.quantile_loss_single(
                    y.clone(), quantile_predictions.clone(), quantile
                )
                total_loss += quantile_loss.mean()  # Mean over samples and features

        total_loss = torch.tensor(total_loss).clone().detach().requires_grad_(True)
        return total_loss

    def calculate_q_risk(self, y, predictions_list, quantiles):
        q_risks = []
        best_quantiles = []
        best_predictions = []  # Store the best predictions for each batch
        for batch_idx in range(y.shape[0]):
            candidates = [
                predictions_list[quantile_idx][batch_idx]
                for quantile_idx in range(len(quantiles))
            ]
            ground_truth = y[batch_idx]

            sum_abs_y = torch.sum(torch.abs(ground_truth))
            best_q_risk = float("inf")
            best_quantile = None
            best_candidate = (
                None  # Store the best candidate prediction for the current batch
            )

            for quantile_idx, candidate in enumerate(candidates):
                q_risk = 0.0
                for quantile in quantiles:
                    quantile_loss = self.quantile_loss_single(
                        ground_truth, candidate, quantile
                    )
                    q_risk += quantile_loss
                q_risk /= sum_abs_y
                q_risk *= 2
                if q_risk < best_q_risk:
                    best_q_risk = q_risk
                    best_quantile = quantiles[quantile_idx]
                    best_candidate = candidate  # Update the best candidate
            q_risks.append(best_q_risk.item())
            best_quantiles.append(best_quantile)
            best_predictions.append(
                best_candidate
            )  # Append the best candidate for the current batch
        return (
            best_quantiles,
            q_risks,
            torch.stack(best_predictions),
        )  # Stack the best predictions tensor
