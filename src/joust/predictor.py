from dataclasses import dataclass
from abc import ABC, abstractmethod

import numpy as np

from joust.dto import Match, Team

@dataclass
class MatchPrediction:
    match: Match
    winner: Team
    loser: Team
    probability: float

    @property
    def odds(self):
        p = self.probability
        return p / (1 - p)
    
    @property
    def min_acceptable_odds(self, margin: float = 0.1):
        return {
            self.winner: 1 / self.probability * (1 + margin),
            self.loser: 1 / (1 - self.probability) * (1 + margin)
        }

class MatchPredictor(ABC):
    def __init__(self) -> None:
        pass

    @abstractmethod
    def predict_winner(self, match: Match) -> MatchPrediction:
        pass

class RandomMatchPredictor(MatchPredictor):
    def predict_winner(self, match: Match) -> MatchPrediction:
        winner, loser = np.random.shuffle([match.left, match.right])
        return MatchPrediction(
            match=match,
            winner=winner,
            loser=loser,
            probability=0.5
        )