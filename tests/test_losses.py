import torch

from cyberworld.models import TemporalJEPAWorldModel
from cyberworld.training.losses import (
    LossWeights,
    compute_loss,
    inverse_frequency_stage_weights,
)


class _StageDataset:
    def __init__(self, samples: list[dict[str, torch.Tensor]]) -> None:
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return self.samples[index]


def test_inverse_frequency_stage_weights_are_train_only_and_balanced() -> None:
    dataset = _StageDataset(
        [
            {
                "future_stages": torch.tensor([0, 0, 1, 2]),
                "future_label_valid": torch.tensor([1.0, 1.0, 1.0, 0.0]),
            },
            {
                "future_stages": torch.tensor([0, 1, 1, 3]),
                "future_label_valid": torch.tensor([1.0, 1.0, 1.0, 0.0]),
            },
        ]
    )

    weights, counts = inverse_frequency_stage_weights(dataset, num_stages=4)

    assert counts.tolist() == [3, 3, 0, 0]
    assert weights.tolist() == [1.0, 1.0, 0.0, 0.0]


def test_stage_cross_entropy_applies_hand_computed_class_weights() -> None:
    logits = torch.tensor([[[0.0, 0.0], [0.0, 0.0]]])
    prediction = type(
        "Prediction",
        (),
        {
            "state_mean": torch.zeros(1, 2, 1),
            "state_log_variance": torch.zeros(1, 2, 1),
            "stage_logits": logits,
            "malicious_logits": torch.zeros(1, 2),
            "compromise_logits": torch.zeros(1, 2),
            "predicted_latents": None,
            "target_latents": None,
        },
    )()
    batch = {
        "future_states": torch.zeros(1, 2, 1),
        "future_availability": torch.ones(1, 2, 1),
        "future_stages": torch.tensor([[0, 1]]),
        "future_label_valid": torch.ones(1, 2),
        "future_malicious": torch.zeros(1, 2),
        "future_compromise": torch.zeros(1, 2),
    }
    loss_weights = LossWeights(0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

    losses = compute_loss(
        prediction,
        batch,
        loss_weights,
        stage_class_weights=torch.tensor([0.5, 2.0]),
    )

    expected = 1.25 * torch.log(torch.tensor(2.0))
    assert torch.isclose(losses.stage, expected)


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
