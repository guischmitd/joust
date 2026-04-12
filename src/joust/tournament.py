from abc import ABC, abstractmethod

from joust.dto import Match, Team
from joust.predictor import MatchPrediction, MatchPredictor
from joust.seeding import recalculate_seeds


class MatchupStrategy(ABC):
    """Base class for matching teams in a round."""

    @abstractmethod
    def get_matchups(self, teams: list[Team], history: list[Match]) -> list[Match]:
        raise NotImplementedError()


class PureSeedMatchup(MatchupStrategy):
    def get_matchups(self, teams: list[Team], history: list[Match]) -> list[Match]:
        assert len(teams) > 0 and len(teams) % 2 == 0, f"Need even >0 teams, got {len(teams)}"
        ordered = sorted(teams, key=lambda t: t.seed)
        half = len(ordered) // 2
        return [Match(a, b) for a, b in zip(ordered[:half], reversed(ordered[half:]))]


class Tournament:
    def __init__(
        self,
        teams: list[Team],
        round_strategies: list[MatchupStrategy],
        match_predictor: MatchPredictor,
    ) -> None:
        self.teams = teams
        self.round_strategies = round_strategies
        self.match_predictor = match_predictor
        self.current_round = 0
        self.match_history: list[Match] = []

    @property
    def active_teams(self) -> list[Team]:
        return [t for t in self.teams if t.wins < 3 and t.losses < 3]

    @property
    def advanced(self) -> list[Team]:
        return [t for t in self.teams if t.wins >= 3]

    @property
    def eliminated(self) -> list[Team]:
        return [t for t in self.teams if t.losses >= 3]

    def tick(self) -> list[MatchPrediction]:
        strategy = self.round_strategies[self.current_round]
        matches = strategy.get_matchups(self.active_teams, self.match_history)
        predictions = [self.match_predictor.predict_winner(m) for m in matches]

        for pred in predictions:
            pred.match.left_wins = pred.winner is pred.match.left
            pred.winner.wins += 1
            pred.loser.losses += 1
            self.match_history.append(pred.match)

        self.current_round += 1
        recalculate_seeds(self.teams, self.match_history)
        return predictions
