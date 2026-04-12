from joust.dto import Match, Team


def difficulty_score(team: Team, history: list[Match]) -> int:
    """Buchholz: sum of (wins - losses) of all opponents faced."""
    score = 0
    for m in history:
        if m.left_wins is None:
            continue
        if m.left is team:
            opp = m.right
        elif m.right is team:
            opp = m.left
        else:
            continue
        score += opp.wins - opp.losses
    return score


def recalculate_seeds(teams: list[Team], history: list[Match]) -> None:
    """Re-seed in place by: W-L record (desc), Buchholz (desc), initial seed (asc)."""
    teams.sort(
        key=lambda t: (
            -(t.wins - t.losses),
            -difficulty_score(t, history),
            t.initial_seed,
        )
    )
    for i, t in enumerate(teams):
        t.seed = i + 1
