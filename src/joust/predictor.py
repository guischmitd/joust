import random
from abc import ABC, abstractmethod
from collections import defaultdict

import numpy as np
import pandas as pd

from joust.dto import Match, MatchPrediction
from joust.features.elo import elo_expected_score
from joust.utils import dataframe_from_matches, matches_from_dataframe


class MatchPredictor(ABC):
    @abstractmethod
    def fit(self, history: list[Match] | pd.DataFrame):
        pass

    @abstractmethod
    def predict_winner(self, match: Match) -> MatchPrediction:
        pass

    def predict_batch(self, matches: list[Match]) -> list[MatchPrediction]:
        return [self.predict_winner(match) for match in matches]

    def _prediction_from_prob(self, match, prob):
        return MatchPrediction(match, round(prob), prob)

    @property
    def updateable(self) -> bool:
        """Dynamically checks if the subclass has overridden the update method."""
        return type(self).update is not MatchPredictor.update

    def update(self, history: list[Match] | pd.DataFrame):
        """Optional method. Subclasses can override this to support incremental updates."""
        raise NotImplementedError(f"Predictor {self.__class__.__name__} does not support incremental updates.")


class RandomMatchPredictor(MatchPredictor):
    def fit(self, history: list[Match] | pd.DataFrame):
        pass

    def predict_winner(self, match: Match) -> MatchPrediction:
        prob = round(random.random())
        return self._prediction_from_prob(match, prob)


class OracleMatchPredictor(MatchPredictor):
    def __init__(self, future_matches: list[Match] | pd.DataFrame) -> None:
        if isinstance(future_matches, pd.DataFrame):
            self.future_matches = matches_from_dataframe(future_matches)
        else:
            self.future_matches = future_matches

        self.future_matches = self.future_matches.copy()

        super().__init__()

    def fit(self, history: list[Match] | pd.DataFrame):
        pass

    def predict_winner(self, match: Match) -> MatchPrediction:
        possible_matches = [m for m in self.future_matches if m == match]

        if len(possible_matches) != 1:
            raise KeyError(
                f"{match.left.name} x {match.right.name}"
                f" failed to fetch a single result from future_matches:\n\n{possible_matches}"
            )

        real_match = possible_matches[0]
        left_wins = real_match.left_wins if real_match.left == match.left else 1 - real_match.left_wins
        pred = self._prediction_from_prob(match, float(left_wins))
        return pred


class EloPredictor(MatchPredictor):
    def __init__(self, k=32) -> None:
        self.k = k
        self.ratings = defaultdict(lambda: 1500)

    def _update_ratings(self, batch: list[Match]):
        for m in batch:
            left, right, left_wins = m.left, m.right, float(m.left_wins)
            scores = {left: left_wins, right: 1.0 - left_wins}
            left_prob = elo_expected_score(self.ratings[left], self.ratings[right])
            expected = {left: left_prob, right: 1.0 - left_prob}
            for team in (left, right):
                self.ratings[team] += self.k * (scores[team] - expected[team])

    def fit(self, history: list[Match] | pd.DataFrame):
        if isinstance(history, pd.DataFrame):
            history = matches_from_dataframe(history)

        self._update_ratings(history)

    def predict_winner(self, match: Match) -> MatchPrediction:
        return self._prediction_from_prob(
            match, elo_expected_score(self.ratings[match.left], self.ratings[match.right])
        )

    def update(self, new_matches: list[Match]):
        self._update_ratings(new_matches)


class GlickoPredictor(MatchPredictor):
    def __init__(self, c=32, freq="W"):
        self.c = c
        self.q = np.log(10) / 400
        self.freq = freq

        self.ratings = defaultdict(lambda: 1500.0)
        self.rds = defaultdict(lambda: 350.0)

    def _g(self, rd):
        return 1 / np.sqrt(1 + 3 * (self.q**2) * (rd**2) / (np.pi**2))

    def _expected_score(self, r, rj, rdj):
        return 1 / (1 + 10 ** (-self._g(rdj) * (r - rj) / 400))

    def _update_ratings(self, batch: list[Match]):
        # Step 1: RD inflation
        for team, rd in self.rds.items():
            self.rds[team] = max(30, min(np.sqrt(rd**2 + self.c**2), 350))

        # Group matches per team
        matches_by_team = defaultdict(list)

        for match in batch:
            matches_by_team[match.left].append((match.right, float(match.left_wins)))
            matches_by_team[match.right].append((match.left, 1.0 - float(match.left_wins)))

        # Step 2: compute updates
        new_ratings = self.ratings.copy()
        new_rds = self.rds.copy()

        for team, team_matches in matches_by_team.items():
            r = self.ratings[team]
            rd = self.rds[team]

            sum_delta = 0.0
            sum_d2 = 0.0

            for opponent, score in team_matches:
                rj = self.ratings[opponent]
                rdj = self.rds[opponent]

                g = self._g(rdj)
                E = self._expected_score(r, rj, rdj)

                sum_delta += g * (score - E)
                sum_d2 += (g**2) * E * (1 - E)

            if sum_d2 == 0:
                continue

            d2 = 1 / (self.q**2 * sum_d2)

            new_ratings[team] = r + self.q / (1 / rd**2 + 1 / d2) * sum_delta

            new_rds[team] = np.sqrt(1 / (1 / rd**2 + 1 / d2))

        self.ratings.update(new_ratings)
        self.rds.update(new_rds)

    def fit(self, history: list[Match] | pd.DataFrame):
        if isinstance(history, list):
            history = dataframe_from_matches(history)

        dates = list(pd.date_range(history["date"].min(), history["date"].max(), freq=self.freq, inclusive="both"))

        if dates[0] >= history["date"].min():
            dates = [pd.to_datetime("1984-01-01"), *dates]

        if dates[-1] < history["date"].max():
            dates.append(pd.to_datetime("2222-02-02"))

        for start_date, end_date in zip(dates[:-1], dates[1:]):
            batch = matches_from_dataframe(
                history[history["date"].dt.date.between(start_date.date(), end_date.date(), inclusive="right")]
            )
            self._update_ratings(batch)

    def update(self, new_matches: list[Match]):
        self._update_ratings(new_matches)

    def predict_winner(self, match: Match) -> MatchPrediction:
        left_rating = self.ratings[match.left]
        right_rating = self.ratings[match.right]

        right_rd = self.rds[match.right]

        left_prob = self._expected_score(left_rating, right_rating, right_rd)

        return self._prediction_from_prob(match, left_prob)


class SklearnMatchPredictor(MatchPredictor):
    def __init__(self, estimator, refit=False) -> None:
        self.estimator = estimator
        self.refit = refit
        super().__init__()

    def fit(self, history: list[Match] | pd.DataFrame):
        if self.refit:
            if isinstance(history, list):
                history = dataframe_from_matches(history)

            self.estimator.fit(history)

    def predict_batch(self, matches: list[Match]) -> list[MatchPrediction]:
        X = dataframe_from_matches(matches)
        probs = self.estimator.predict_proba(X)[:, -1]
        return [self._prediction_from_prob(m, p) for m, p in (matches, probs)]

    def predict_winner(self, match: Match) -> MatchPrediction:
        return super().predict_winner(match)
