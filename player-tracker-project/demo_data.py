"""Placeholder details for demonstrations.

Everything produced here is SIMULATED. It is not measured from the video, it is shown with a
"(demo)" label in the app, and it never feeds the fit score.
"""
import zlib

import numpy as np

POSITIONS = [
    "Right winger", "Left winger", "Striker", "Attacking midfielder",
    "Central midfielder", "Defensive midfielder", "Full-back", "Centre-back",
]
_POSITION_ODDS = [0.17, 0.17, 0.12, 0.10, 0.14, 0.08, 0.14, 0.08]

_SURNAMES = [
    "Almeida", "Brandt", "Castillo", "Dumont", "Easton", "Ferrara", "Gallego", "Hartmann", "Ibarra",
    "Jansen", "Kowalski", "Lindqvist", "Moreau", "Novak", "Okafor", "Petrov", "Quinn", "Rinaldi",
    "Sandoval", "Tanaka", "Ulrich", "Vasquez", "Weber", "Yilmaz", "Zielinski",
]


def simulate_player(seed_text):
    """A repeatable made-up name, position, shirt number and passing figures for one player."""
    rng = np.random.default_rng(zlib.crc32(seed_text.encode()))
    initial = chr(65 + int(rng.integers(0, 26)))
    return {
        "name": f"{initial}. {_SURNAMES[int(rng.integers(len(_SURNAMES)))]}",
        "position": str(rng.choice(POSITIONS, p=_POSITION_ODDS)),
        "shirt": str(int(rng.integers(2, 30))),
        "pass_pct": float(round(rng.uniform(68, 92), 0)),
        "prog_passes": float(round(rng.uniform(2, 9), 1)),
    }
