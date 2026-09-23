import torch

from cyberworld.training.metrics import MetricAccumulator


def test_stage_accuracy_is_null_when_all_labels_are_invalid() -> None:
    accumulator = MetricAccumulator()
    accumulator.update(
        loss=torch.tensor(2.0),
        state_mean=torch.zeros(1, 2, 3),
        state_target=torch.zeros(1, 2, 3),
        state_mask=torch.ones(1, 2, 3),
        stage_logits=torch.zeros(1, 2, 9),
        stage_target=torch.full((1, 2), 8, dtype=torch.int64),
        label_valid=torch.zeros(1, 2),
    )

    metrics = accumulator.compute()

    assert metrics["stage_accuracy"] is None
    assert metrics["stage_valid_count"] == 0
    assert metrics["stage_accuracy_status"] == "excluded_no_valid_labels"