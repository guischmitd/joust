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
        wins_to_advance: int = 3,
        losses_to_drop: int = 3,
    ) -> None:
        self.teams = teams
        self.round_strategies = round_strategies
        self.match_predictor = match_predictor
        self.current_round = 0
        self.match_history: list[Match] = []

        assert wins_to_advance > 0, (
            f"wins_to_advance must be strictly positive (received {wins_to_advance})"
        )
        assert losses_to_drop > 0, (
            f"losses_to_drop must be strictly positive (received {losses_to_drop})"
        )

        self.wins_to_advance = wins_to_advance
        self.losses_to_drop = losses_to_drop

    @property
    def active_teams(self) -> list[Team]:
        return [
            t
            for t in self.teams
            if t.wins < self.wins_to_advance and t.losses < self.losses_to_drop
        ]

    @property
    def advanced(self) -> list[Team]:
        return [t for t in self.teams if t.wins >= self.wins_to_advance]

    @property
    def eliminated(self) -> list[Team]:
        return [t for t in self.teams if t.losses >= self.losses_to_drop]

    def tick(self) -> list[MatchPrediction]:
        strategy = self.round_strategies[self.current_round]
        matches = strategy.get_matchups(self.active_teams, self.match_history)
        predictions = [self.match_predictor.predict_winner(m) for m in matches]

        for pred in predictions:
            pred.match.left_wins = pred.left_wins
            pred.winner.wins += 1
            pred.loser.losses += 1
            self.match_history.append(pred.match)

        self.current_round += 1
        recalculate_seeds(self.teams, self.match_history)
        return predictions
