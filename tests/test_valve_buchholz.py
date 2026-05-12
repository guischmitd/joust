from joust.buchholz import ValveBuchholzMatchup
from joust.dto import Contestant, Match, Team


def _make_contestant(name: str, seed: int) -> Contestant:
    return Contestant(team=Team(name), seed=seed)


def compare_matchups(a: list[Match], b: list[Match]) -> bool:
    if len(a) != len(b):
        return False
    return all(m in b for m in a)


def test_round3_no_history():
    t = [
        _make_contestant("A", seed=1),
        _make_contestant("B", seed=2),
        _make_contestant("C", seed=3),
        _make_contestant("D", seed=4),
    ]

    previous_matches: list[Match] = []

    s = ValveBuchholzMatchup(3)
    expected = [Match(t[0].team, t[-1].team), Match(t[1].team, t[-2].team)]

    registry = {c.team: c for c in t}
    matchups = s.get_matchups(t, previous_matches, registry)
    assert compare_matchups(matchups, expected)


def test_round3_one_rematch():
    t = [
        _make_contestant("A", seed=1),
        _make_contestant("B", seed=2),
        _make_contestant("C", seed=3),
        _make_contestant("D", seed=4),
    ]

    previous_matches = [Match(t[0].team, t[-1].team)]
    expected = [Match(t[0].team, t[2].team), Match(t[1].team, t[3].team)]
    s = ValveBuchholzMatchup(3)
    registry = {c.team: c for c in t}
    matchups = s.get_matchups(t, previous_matches, registry)
    assert compare_matchups(matchups, expected)


def test_round3_backtrack():
    t = [
        _make_contestant("A", seed=1),
        _make_contestant("B", seed=2),
        _make_contestant("C", seed=3),
        _make_contestant("D", seed=4),
        _make_contestant("E", seed=5),
        _make_contestant("F", seed=6),
        _make_contestant("G", seed=7),
        _make_contestant("H", seed=8),
    ]

    previous_matches = [
        Match(t[0].team, t[7].team),  # 1-8
        Match(t[1].team, t[6].team),  # 2-7
        Match(t[2].team, t[5].team),  # 3-6
        Match(t[3].team, t[4].team),  # 4-5
        Match(t[1].team, t[7].team),  # 2-8
        Match(t[2].team, t[7].team),  # 3-8
    ]

    expected = [
        Match(t[0].team, t[6].team),  # 1v7
        Match(t[1].team, t[5].team),  # 2v6
        Match(t[2].team, t[4].team),  # 3v5
        Match(t[3].team, t[7].team),  # 4v8
    ]

    s = ValveBuchholzMatchup(3)
    registry = {c.team: c for c in t}
    matchups = s.get_matchups(t, previous_matches, registry)
    assert compare_matchups(matchups, expected)
