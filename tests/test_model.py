import pytest
import torch

from cyberworld.models import (
    TemporalConvolutionalWorldModel,
    TemporalJEPAWorldModel,
    TemporalTransformerWorldModel,
    WorldModel,
)


@pytest.mark.parametrize(
    "model_type",
    [
        WorldModel,
        TemporalJEPAWorldModel,
        TemporalTransformerWorldModel,
        TemporalConvolutionalWorldModel,
    ],
)
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


def test_transformer_respects_availability_mask_and_rolls_out_multiple_steps() -> None:
    model = TemporalTransformerWorldModel(
        feature_dim=4,
        num_stages=9,
        projection_dim=8,
        hidden_dim=8,
        num_layers=2,
        dropout=0.0,
        min_log_variance=-8.0,
        max_log_variance=5.0,
    )
    history = torch.randn(2, 6, 4)
    masked = torch.ones_like(history)
    masked[:, 0] = 0
    output = model.rollout(history, masked, 4)
    assert output.malicious_logits.shape == (2, 4)
    assert torch.isfinite(output.state_mean).all()


def test_tcn_is_causal_and_rolls_out_multiple_steps() -> None:
    model = TemporalConvolutionalWorldModel(
        feature_dim=4,
        num_stages=9,
        projection_dim=8,
        hidden_dim=8,
        num_layers=3,
        dropout=0.0,
        min_log_variance=-8.0,
        max_log_variance=5.0,
    )
    model.eval()
    sequence = torch.randn(1, 6, 8)
    changed_future = sequence.clone()
    changed_future[:, 4:] += 100.0
    first = model._transform(sequence[:, :4])
    second = model._transform(changed_future[:, :4])
    assert torch.allclose(first, second, atol=1e-6)
    output = model.rollout(torch.randn(2, 6, 4), torch.ones(2, 6, 4), 4)
    assert output.stage_logits.shape == (2, 4, 9)
