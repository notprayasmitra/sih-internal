from cyberworld.labels import AttackStage, is_compromise, is_reliably_malicious


def test_operational_targets_are_distinct() -> None:
    assert is_reliably_malicious(AttackStage.IMPACT)
    assert not is_compromise(AttackStage.IMPACT)
    assert is_compromise(AttackStage.LATERAL_MOVEMENT)
    assert not is_reliably_malicious(AttackStage.UNKNOWN)
