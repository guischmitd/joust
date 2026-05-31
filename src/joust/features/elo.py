from collections import defaultdict

import pandas as pd

from joust.features import FeatureSet


def elo_expected_score(left_elo, right_elo):
    diff = right_elo - left_elo
    return 1 / (1 + (10 ** (diff / 400)))


class EloFeatureSet(FeatureSet):
    def __init__(self, k=32, freq="W", **kwargs) -> None:
        self.elos = defaultdict(lambda: 1500)
        self.k = k
        self.freq = freq
        super().__init__()

    def update(self, new_data: pd.DataFrame):
        for _, row in new_data.iterrows():
            left, right, left_wins = row["left"], row["right"], row["left_wins"]
            scores = {left: left_wins, right: 1.0 - left_wins}
            left_prob = elo_expected_score(self.elos[left], self.elos[right])
            expected = {left: left_prob, right: 1 - left_prob}
            for team in (left, right):
                self.elos[team] += self.k * (scores[team] - expected[team])

    def get_elos_for_match(self, match_series: pd.Series):
        left, right = match_series["left"], match_series["right"]
        return pd.Series(
            {
                "elo_left": self.elos[left],
                "elo_right": self.elos[right],
                "elo_diff": self.elos[left] - self.elos[right],
                "elo_prob": elo_expected_score(
                    self.elos[left],
                    self.elos[right],
                ),
            }
        )

    def get_features(self, X: pd.DataFrame):
        return X.apply(self.get_elos_for_match, axis=1)
