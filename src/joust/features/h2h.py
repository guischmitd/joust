from collections import defaultdict

import pandas as pd

from joust.features import FeatureSet


class H2HFeatureSet(FeatureSet):
    def __init__(self, **kwargs) -> None:
        self.h2h = defaultdict(lambda: {"encounters": 0, "wins": defaultdict(int)})

        super().__init__()

    def update(self, new_data: pd.DataFrame):
        for _, row in new_data.iterrows():
            left, right, winner = row["left"], row["right"], row["winner_name"]

            key = tuple(sorted([left, right]))

            self.h2h[key]["encounters"] += 1
            self.h2h[key]["wins"][winner] += 1

    def _get_feats_for_match(self, row):
        left, right = row["left"], row["right"]
        key = tuple(sorted([left, right]))

        data = self.h2h.get(key, None)

        if data is None:
            return pd.Series(
                {
                    "h2h_encounters": 0,
                    "h2h_win_diff": 0,
                    "h2h_left_rel_diff": 0.0,
                }
            )

        encounters = data["encounters"]
        lw = data["wins"].get(left, 0)
        rw = data["wins"].get(right, 0)

        return pd.Series(
            {
                "h2h_encounters": encounters,
                "h2h_win_diff": lw - rw,
                "h2h_left_rel_diff": (lw - rw) / encounters if encounters else 0.0,
            }
        )

    def get_features(self, X: pd.DataFrame):
        return X.apply(self._get_feats_for_match, axis=1)
