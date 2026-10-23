import math
from collections import Counter
from typing import Iterable


def wilson(successes: int, trials: int) -> tuple[float, float] | None:
    if type(successes) is not int or type(trials) is not int or not 0 <= successes <= trials:
        raise ValueError('invalid success counts')
    if trials == 0:
        return None
    z = 1.959963984540054
    rate = successes / trials
    denominator = 1 + z * z / trials
    center = (rate + z * z / (2 * trials)) / denominator
    width = z * math.sqrt(rate * (1 - rate) / trials + z * z / (4 * trials * trials)) / denominator
    return max(0., center - width), min(1., center + width)


def metrics(trials: Iterable[dict]) -> dict:
    rows = list(trials)
    successes = sum(row['success'] is True for row in rows)
    return {'trials': len(rows), 'successes': successes,
            'success_rate': successes / len(rows) if rows else None,
            'wilson_95': wilson(successes, len(rows)),
            'wrong_color_picks': sum(row.get('failure_reason') == 'wrong_color' for row in rows),
            'failures': dict(Counter(row.get('failure_reason') for row in rows if row['success'] is not True))}
