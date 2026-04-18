from joust.dto import Match, Team
from joust.predictor import MatchPrediction, RandomMatchPredictor
from joust.seeding import difficulty_score, recalculate_seeds
from joust.tournament import PureSeedMatchup, Tournament


def _make_teams(n=16):
    return [Team(name=f"Team {i + 1}", seed=i + 1) for i in range(n)]


# --- dto ---


def test_match_defaults():
    a, b = Team("A", 1), Team("B", 2)
    m = Match(a, b)
    assert m.left_wins is None
    assert m.best_of == 1
    assert m.winner is None and m.loser is None


def test_match_winner_loser():
    a, b = Team("A", 1), Team("B", 2)
    m = Match(a, b, left_wins=True)
    assert m.winner is a and m.loser is b
    m2 = Match(a, b, left_wins=False)
    assert m2.winner is b and m2.loser is a


def test_team_initial_seed_preserved():
    t = Team("X", seed=3)
    assert t.initial_seed == 3
    t.seed = 1
    assert t.initial_seed == 3


# --- predictor ---


def test_random_predictor_returns_valid_prediction():
    a, b = Team("A", 1), Team("B", 2)
    m = Match(a, b)
    pred = RandomMatchPredictor().predict_winner(m)
    assert isinstance(pred, MatchPrediction)
    assert pred.winner in (a, b)
    assert pred.loser in (a, b)
    assert pred.winner is not pred.loser
    assert pred.probability == 0.5


def test_min_acceptable_odds_is_callable():
    a, b = Team("A", 1), Team("B", 2)
    pred = MatchPrediction(Match(a, b), winner=a, loser=b, probability=0.6)
    odds = pred.min_acceptable_odds(margin=0.05)
    assert "left" in odds and "right" in odds


# --- seeding ---


def test_difficulty_score_basic():
    a, b, c = Team("A", 1), Team("B", 2), Team("C", 3)
    b.wins, b.losses = 2, 0
    c.wins, c.losses = 1, 1
    history = [
        Match(a, b, left_wins=False),  # A lost to B
        Match(a, c, left_wins=True),  # A beat C
    ]
    # opponents of A: B(2-0) + C(1-1) => (2-0)+(1-1) = 2
    assert difficulty_score(a, history) == 2


def test_recalculate_seeds():
    a = Team("A", seed=1)
    b = Team("B", seed=2)
    a.wins, a.losses = 0, 1
    b.wins, b.losses = 1, 0
    recalculate_seeds([a, b], [])
    assert b.seed < a.seed  # b should be seeded higher (lower number)


# --- PureSeedMatchup ---


def test_pure_seed_matchup_pairs_correctly():
    teams = _make_teams(4)
    matches = PureSeedMatchup().get_matchups(teams, [])
    pairs = {(m.left.seed, m.right.seed) for m in matches}
    assert pairs == {(1, 4), (2, 3)}


# --- Tournament integration ---


def test_tick_updates_state():
    teams = _make_teams(4)
    t = Tournament(teams, [PureSeedMatchup()], RandomMatchPredictor())
    preds = t.tick()
    assert len(preds) == 2
    assert t.current_round == 1
    assert len(t.match_history) == 2
    total_wins = sum(tm.wins for tm in teams)
    total_losses = sum(tm.losses for tm in teams)
    assert total_wins == 2 and total_losses == 2


def test_three_round_tournament_produces_results():
    """Run 3 rounds of pure-seed with 8 teams. After 3 rounds every team has 3 matches."""
    teams = _make_teams(8)
    strategies = [PureSeedMatchup()] * 3
    t = Tournament(teams, strategies, RandomMatchPredictor())
    for _ in range(3):
        t.tick()

    assert len(t.match_history) == 12  # 4 matches/round * 3 rounds
    for tm in teams:
        assert tm.wins + tm.losses == 3


def test_seeds_update_after_tick():
    teams = _make_teams(4)
    t = Tournament(teams, [PureSeedMatchup()], RandomMatchPredictor())
    t.tick()
    # Winners should have lower seeds than losers after reseeding
    winners = [tm for tm in teams if tm.wins == 1]
    losers = [tm for tm in teams if tm.losses == 1]
    assert all(w.seed < loser.seed for w in winners for loser in losers)
