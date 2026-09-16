"""Conservative cross-dataset operational labels."""

from enum import IntEnum


class AttackStage(IntEnum):
    NORMAL = 0
    RECONNAISSANCE = 1
    INITIAL_ACCESS = 2
    LATERAL_MOVEMENT = 3
    COMMAND_AND_CONTROL = 4
    EXFILTRATION = 5
    IMPACT = 6
    OTHER_MALICIOUS = 7
    UNKNOWN = 8


COMPROMISE_STAGES = frozenset(
    {
        AttackStage.INITIAL_ACCESS,
        AttackStage.LATERAL_MOVEMENT,
        AttackStage.COMMAND_AND_CONTROL,
        AttackStage.EXFILTRATION,
    }
)


def is_reliably_malicious(stage: int | AttackStage) -> bool:
    """Return whether a label is malicious and usable as supervised truth."""
    value = AttackStage(stage)
    return value not in {AttackStage.NORMAL, AttackStage.UNKNOWN}


def is_compromise(stage: int | AttackStage) -> bool:
    """Return whether the stage satisfies the operational compromise target."""
    return AttackStage(stage) in COMPROMISE_STAGES
