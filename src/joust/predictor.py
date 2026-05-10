import random
from abc import ABC, abstractmethod

import pandas as pd

from joust.dto import Match, MatchPrediction
from joust.utils import matches_from_dataframe


class MatchPredictor(ABC):
    @abstractmethod
    def predict_winner(self, match: Match) -> MatchPrediction:
        pass

    def predict_batch(self, matches: list[Match]) -> list[MatchPrediction]:
        return [self.predict_winner(match) for match in matches]

    def _prediction_from_prob(self, match, prob):
        winner = match.left if round(prob) else match.right
        loser = match.left if (not round(prob)) else match.right
        return MatchPrediction(match, winner, loser, prob)


class RandomMatchPredictor(MatchPredictor):
    def predict_winner(self, match: Match) -> MatchPrediction:
        pair = [match.left, match.right]
        random.shuffle(pair)
        winner, loser = pair
        return MatchPrediction(match=match, winner=winner, loser=loser, probability=0.5)


class OracleMatchPredictor(MatchPredictor):
    def __init__(self, future_matches: list[Match] | pd.DataFrame) -> None:
        if isinstance(future_matches, pd.DataFrame):
            self.future_matches = matches_from_dataframe(future_matches)
        else:
            self.future_matches = future_matches

        self.future_matches = self.future_matches.copy()

        super().__init__()

    def predict_winner(self, match: Match) -> MatchPrediction:
        possible_matches = [m for m in self.future_matches if m == match]

        if len(possible_matches) != 1:
            raise KeyError(
                f"{match.left.name} x {match.right.name}"
                f" failed to fetch a single result from future_matches:\n\n{possible_matches}"
            )

        real_match = possible_matches[0]
        winner = match.left if real_match.left_wins else match.right
        loser = match.right if real_match.left_wins else match.left

        pred = MatchPrediction(match, winner=winner, loser=loser, probability=1.0)
        return pred


class SklearnMatchPredictor(MatchPredictor):
    def __init__(self, estimator) -> None:
        self.estimator = estimator
        super().__init__()

    def predict_batch(self, matches: list[Match]) -> list[MatchPrediction]:
        X = pd.concat([dict(m) for m in matches])
        probs = self.estimator.predict_proba(X)
        return [self._prediction_from_prob(m, p) for m, p in (matches, probs)]

    def predict_winner(self, match: Match) -> MatchPrediction:
        return super().predict_winner(match)
