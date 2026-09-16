import torch

from cyberworld.models import TemporalJEPAWorldModel
from cyberworld.training.losses import LossWeights, compute_loss


def test_multitask_jepa_loss_is_finite() -> None:
    model = TemporalJEPAWorldModel(
        feature_dim=6,
        num_stages=9,
        projection_dim=10,
        latent_dim=8,
        hidden_dim=12,
        num_layers=1,
        dropout=0.0,
        min_log_variance=-8.0,
        max_log_variance=5.0,
    )
    history = torch.randn(5, 4, 6)
    future = torch.randn(5, 3, 6)
    prediction = model.rollout(
        history,
        torch.ones_like(history),
        3,
        teacher_states=future,
        teacher_availability=torch.ones_like(future),
    )
    batch = {
        "future_states": future,
        "future_availability": torch.ones_like(future),
        "future_stages": torch.randint(0, 8, (5, 3)),
        "future_label_valid": torch.ones(5, 3),
        "future_malicious": torch.ones(5, 3),
        "future_compromise": torch.zeros(5, 3),
    }
    weights = LossWeights(1.0, 0.5, 0.5, 0.5, 0.25, 25.0, 25.0, 1.0)
    losses = compute_loss(prediction, batch, weights)
    assert torch.isfinite(losses.total)
    assert losses.jepa.item() > 0
    losses.total.backward()
