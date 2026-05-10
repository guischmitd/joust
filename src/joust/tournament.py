from abc import ABC, abstractmethod

from joust.dto import Contestant, Match
from joust.predictor import MatchPrediction, MatchPredictor, RandomMatchPredictor


class MatchupStrategy(ABC):
    """Base class for matching teams in a round."""

    @abstractmethod
    def get_matchups(self, contestants: list[Contestant], match_history: list[Match]) -> list[Match]:
        raise NotImplementedError()

    def update_seeds(self, contestants: list[Contestant], match_history: list[Match]) -> None:
        pass


class PureSeedMatchup(MatchupStrategy):
    """Common matchup strategy for bracket style play (playoffs)"""

    def get_matchups(self, contestants: list[Contestant], match_history: list[Match]) -> list[Match]:
        assert len(contestants) > 0 and len(contestants) % 2 == 0, f"Need even >0 contestants, got {len(contestants)}"
        ordered = sorted(contestants, key=lambda c: c.seed)
        half = len(ordered) // 2
        return [Match(a.team, b.team) for a, b in zip(ordered[:half], reversed(ordered[half:]))]


class Tournament:
    def __init__(
        self,
        contestants: list[Contestant],
        round_strategies: list[MatchupStrategy],
        match_predictor: MatchPredictor | None = None,
        wins_to_advance: int = 3,
        losses_to_drop: int = 3,
    ) -> None:
        self.contestants = contestants
        self.round_strategies = round_strategies
        self.match_predictor = match_predictor or RandomMatchPredictor()
        self.current_round = 0
        self.match_history: list[Match] = []

        assert wins_to_advance > 0, f"wins_to_advance must be strictly positive (received {wins_to_advance})"
        assert losses_to_drop > 0, f"losses_to_drop must be strictly positive (received {losses_to_drop})"

        self.wins_to_advance = wins_to_advance
        self.losses_to_drop = losses_to_drop

    @property
    def active_contestants(self) -> list[Contestant]:
        return [c for c in self.contestants if c.wins < self.wins_to_advance and c.losses < self.losses_to_drop]

    @property
    def advanced(self) -> list[Contestant]:
        return [c for c in self.contestants if c.wins >= self.wins_to_advance]

    @property
    def eliminated(self) -> list[Contestant]:
        return [c for c in self.contestants if c.losses >= self.losses_to_drop]

    def tick(self) -> list[MatchPrediction]:
        strategy = self.round_strategies[self.current_round]
        strategy.update_seeds(self.contestants, self.match_history)
        matches = strategy.get_matchups(self.active_contestants, self.match_history)
        predictions = [self.match_predictor.predict_winner(m) for m in matches]

        team_to_contestant = {c.team: c for c in self.contestants}
        for pred in predictions:
            pred.match.left_wins = pred.left_wins
            team_to_contestant[pred.winner].wins += 1
            team_to_contestant[pred.loser].losses += 1
            self.match_history.append(pred.match)

        self.current_round += 1
        return predictions

    def run_to_completion(self) -> "Tournament":
        while self.active_contestants:
            self.tick()
        return self

    @property
    def standings(self) -> list[Contestant]:
        return sorted(self.contestants, key=lambda c: c.seed)
