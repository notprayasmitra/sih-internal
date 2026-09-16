import pytest
import torch

from cyberworld.models import TemporalJEPAWorldModel, WorldModel


@pytest.mark.parametrize("model_type", [WorldModel, TemporalJEPAWorldModel])
def test_rollout_shapes_and_gradients(model_type) -> None:
    kwargs = dict(
        feature_dim=8,
        num_stages=9,
        projection_dim=12,
        hidden_dim=16,
        num_layers=1,
        dropout=0.0,
        min_log_variance=-8.0,
        max_log_variance=5.0,
    )
    if model_type is TemporalJEPAWorldModel:
        kwargs["latent_dim"] = 10
    model = model_type(**kwargs)
    history = torch.randn(4, 5, 8)
    mask = torch.ones_like(history)
    future = torch.randn(4, 3, 8)
    output = model.rollout(
        history,
        mask,
        3,
        teacher_states=future,
        teacher_availability=torch.ones_like(future),
    )
    assert output.state_mean.shape == (4, 3, 8)
    assert output.state_log_variance.shape == (4, 3, 8)
    assert output.stage_logits.shape == (4, 3, 9)
    assert output.malicious_logits.shape == (4, 3)
    loss = output.state_mean.square().mean() + output.stage_logits.square().mean()
    loss.backward()
    assert any(parameter.grad is not None for parameter in model.parameters())
    if model_type is TemporalJEPAWorldModel:
        assert output.predicted_latents is not None
        assert output.target_latents is not None
