import random
from abc import ABC, abstractmethod
from dataclasses import dataclass

from joust.dto import Match, Team


@dataclass
class MatchPrediction:
    match: Match
    winner: Team
    loser: Team
    probability: float

    @property
    def odds(self) -> float:
        p = self.probability
        return p / (1 - p)

    def min_acceptable_odds(self, margin: float = 0.1) -> dict[Team, float]:
        return {
            self.winner: 1 / self.probability * (1 + margin),
            self.loser: 1 / (1 - self.probability) * (1 + margin),
        }


class MatchPredictor(ABC):
    @abstractmethod
    def predict_winner(self, match: Match) -> MatchPrediction:
        pass


class RandomMatchPredictor(MatchPredictor):
    def predict_winner(self, match: Match) -> MatchPrediction:
        pair = [match.left, match.right]
        random.shuffle(pair)
        winner, loser = pair
        return MatchPrediction(match=match, winner=winner, loser=loser, probability=0.5)
