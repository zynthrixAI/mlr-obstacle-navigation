"""Seed ranges. Disjointness is asserted at import time."""
SUPERVISED = range(100_000, 100_300)   # 100 maps per type
RL_TRAIN = range(0, 10_000)
RL_EVAL = range(200_000, 200_150)      # 50 maps per type
MAP_TYPES = ("open", "narrow", "cluttered")

_sets = [set(SUPERVISED), set(RL_TRAIN), set(RL_EVAL)]
for i in range(3):
    for j in range(i + 1, 3):
        assert not (_sets[i] & _sets[j]), "seed ranges overlap"


def map_type_for(seed: int) -> str:
    """Deterministic, balanced map type per seed."""
    return MAP_TYPES[seed % 3]
