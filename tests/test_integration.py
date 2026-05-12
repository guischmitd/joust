from statistics import mean

import pytest

from joust.buchholz import buchholz_difficulty_score, recalculate_buchholz_seeds
from joust.dto import Contestant, Match, Team
from joust.predictor import MatchPrediction, RandomMatchPredictor
from joust.tournament import PureSeedMatchup, Tournament


def _make_contestants(n=16) -> list[Contestant]:
    return [Contestant(team=Team(name=f"Team {i + 1}"), seed=i + 1) for i in range(n)]


def _make_registry(contestants: list[Contestant]) -> dict[Team, Contestant]:
    return {c.team: c for c in contestants}


# --- dto ---


def test_match_defaults():
    a, b = Team("A"), Team("B")
    m = Match(a, b)
    assert m.left_wins is None
    assert m.best_of == 1
    assert m.winner is None and m.loser is None


def test_match_winner_loser():
    a, b = Team("A"), Team("B")
    m = Match(a, b, left_wins=True)
    assert m.winner is a and m.loser is b
    m2 = Match(a, b, left_wins=False)
    assert m2.winner is b and m2.loser is a


def test_contestant_initial_seed_preserved():
    c = Contestant(team=Team("X"), seed=3)
    assert c.initial_seed == 3
    c.seed = 1
    assert c.initial_seed == 3


# --- predictor ---


def test_random_predictor_returns_valid_prediction():
    a, b = Team("A"), Team("B")
    m = Match(a, b)

    preds = [RandomMatchPredictor().predict_winner(m) for _ in range(10_000)]
    probs = [pred.probability for pred in preds]
    assert isinstance(preds[0], MatchPrediction)
    assert all([pred.predicted_winner in (a, b) for pred in preds])
    assert all([pred.predicted_loser in (a, b) for pred in preds])
    assert all([pred.predicted_winner is not pred.predicted_loser for pred in preds])
    pytest.approx(0.5, mean(probs))


def test_min_acceptable_odds_is_callable():
    a, b = Team("A"), Team("B")
    pred = MatchPrediction(Match(a, b, left_wins=True), left_wins=True, probability=0.6)
    odds = pred.min_acceptable_odds(margin=0.05)
    assert "left" in odds and "right" in odds


# --- seeding (now in buchholz) ---


def test_difficulty_score_basic():
    a, b, c = Team("A"), Team("B"), Team("C")
    ca = Contestant(team=a, seed=1)
    cb = Contestant(team=b, seed=2, wins=2, losses=0)
    cc = Contestant(team=c, seed=3, wins=1, losses=1)
    registry = _make_registry([ca, cb, cc])
    history = [
        Match(a, b, left_wins=False),  # A lost to B
        Match(a, c, left_wins=True),  # A beat C
    ]
    # opponents of A: B(2-0) + C(1-1) => (2-0)+(1-1) = 2
    assert buchholz_difficulty_score(ca, history, registry) == 2


def test_recalculate_seeds():
    a = Contestant(team=Team("A"), seed=1, wins=0, losses=1)
    b = Contestant(team=Team("B"), seed=2, wins=1, losses=0)
    registry = _make_registry([a, b])
    recalculate_buchholz_seeds([a, b], [], registry)
    assert b.seed < a.seed  # b should be seeded higher (lower number)


# --- PureSeedMatchup ---


def test_pure_seed_matchup_pairs_correctly():
    contestants = _make_contestants(4)
    registry = _make_registry(contestants)
    matches = PureSeedMatchup().get_matchups(contestants, [], registry)
    seed_of = {c.team.name: c.seed for c in contestants}
    pairs = {(seed_of[m.left.name], seed_of[m.right.name]) for m in matches}
    assert pairs == {(1, 4), (2, 3)}


# --- Tournament integration ---


def test_tick_updates_state():
    contestants = _make_contestants(4)
    t = Tournament(contestants, [PureSeedMatchup()], RandomMatchPredictor())
    preds = t.tick()
    assert len(preds) == 2
    assert t.current_round == 1
    assert len(t.match_history) == 2
    total_wins = sum(c.wins for c in contestants)
    total_losses = sum(c.losses for c in contestants)
    assert total_wins == 2 and total_losses == 2


def test_three_round_tournament_produces_results():
    """Run 3 rounds of pure-seed with 8 teams. After 3 rounds every team has 3 matches."""
    contestants = _make_contestants(8)
    strategies = [PureSeedMatchup()] * 3
    t = Tournament(contestants, strategies, RandomMatchPredictor())
    for _ in range(3):
        t.tick()

    assert len(t.match_history) == 12  # 4 matches/round * 3 rounds
    for c in contestants:
        assert c.wins + c.losses == 3


def test_seeds_unchanged_with_pure_seed():
    """PureSeedMatchup does not re-seed; seeds should remain as initially assigned."""
    contestants = _make_contestants(4)
    original_seeds = {c.team: c.seed for c in contestants}
    t = Tournament(contestants, [PureSeedMatchup()], RandomMatchPredictor())
    t.tick()
    for c in contestants:
        assert c.seed == original_seeds[c.team]
