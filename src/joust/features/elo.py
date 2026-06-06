from collections import defaultdict
from collections.abc import Iterable
from typing import Literal

import numpy as np
import pandas as pd

from joust.features import FeatureSet


def elo_expected_score(left_elo, right_elo):
    diff = right_elo - left_elo
    return 1 / (1 + (10 ** (diff / 400)))


class EloFeatureSet(FeatureSet):
    def __init__(
        self,
        k=32,
        freq="W",
        tier_filter: int | None = None,
        tier_filter_mode: Literal["strict", "relaxed"] = "strict",
        **kwargs,
    ) -> None:
        self.elos = defaultdict(lambda: 1500)
        self.k = k
        self.freq = freq
        self.tier_filter = tier_filter
        self.tier_filter_mode = tier_filter_mode

        self.suffix = "" if tier_filter is None else f"_{tier_filter_mode}_tier{tier_filter}"
        super().__init__()

    def update(self, new_data: pd.DataFrame):
        if self.tier_filter:
            if self.tier_filter_mode == "strict":
                filtered_data = new_data[
                    (new_data["left_tier"] <= self.tier_filter) & (new_data["right_tier"] <= self.tier_filter)
                ]
            elif self.tier_filter_mode == "relaxed":
                filtered_data = new_data[
                    (new_data["left_tier"] <= self.tier_filter) | (new_data["right_tier"] <= self.tier_filter)
                ]
            else:
                raise ValueError(f"Invalid tier filtering mode {self.filter_tier_mode}")
        else:
            filtered_data = new_data

        for _, row in filtered_data.iterrows():
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
                f"elo_left{self.suffix}": self.elos[left],
                f"elo_right{self.suffix}": self.elos[right],
                f"elo_diff{self.suffix}": self.elos[left] - self.elos[right],
                f"elo_prob{self.suffix}": elo_expected_score(
                    self.elos[left],
                    self.elos[right],
                ),
            }
        )

    def get_features(self, X: pd.DataFrame):
        return X.apply(self.get_elos_for_match, axis=1)


class GlickoFeatureSet(FeatureSet):
    def __init__(
        self,
        c=16,
        freq="W",
        tier_filter: int | None = None,
        tier_filter_mode: Literal["strict", "relaxed"] = "strict",
        **kwargs,
    ) -> None:
        self.ratings = defaultdict(lambda: 1500)
        self.rds = defaultdict(lambda: 350)
        self.c = c
        self.q = np.log(10) / 400
        self.freq = freq

        self.tier_filter = tier_filter
        self.tier_filter_mode = tier_filter_mode

        self.suffix = "" if tier_filter is None else f"_{tier_filter_mode}_tier{tier_filter}"
        super().__init__()

    def __g(self, rd):
        return 1 / np.sqrt(1 + 3 * (self.q**2) * (rd**2) / (np.pi**2))

    def _expected_scores(self, left, right):
        rd_left, rd_right = self.rds[left], self.rds[right]
        norm_rds = np.sqrt(rd_left**2 + rd_right**2)
        rating_diff = self.ratings[left] - self.ratings[right]

        expected = {left: 1 / (1 + 10 ** (-self.__g(norm_rds) * rating_diff / 400))}
        expected[right] = 1.0 - expected[left]
        return expected

    def __E(self, r, rj, rdj):
        return 1 / (1 + 10 ** (-self.__g(rdj) * (r - rj) / 400))

    def update(self, new_data: pd.DataFrame):
        for team, rd in self.rds.items():
            # Glicko step 1
            self.rds[team] = max(30, min(np.sqrt(rd**2 + self.c**2), 350))

        # Glicko step 2 (compute all then apply)
        new_ratings = self.ratings.copy()
        new_rds = self.rds.copy()

        if self.tier_filter:
            if self.tier_filter_mode == "strict":
                filtered_data = new_data[
                    (new_data["left_tier"] == self.tier_filter) & (new_data["right_tier"] == self.tier_filter)
                ]
            elif self.tier_filter_mode == "relaxed":
                filtered_data = new_data[
                    (new_data["left_tier"] == self.tier_filter) | (new_data["right_tier"] == self.tier_filter)
                ]
            else:
                raise ValueError(f"Invalid tier filtering mode {self.filter_tier_mode}")
        else:
            filtered_data = new_data

        unique_teams = set(filtered_data["left"].tolist() + filtered_data["right"].tolist())

        for team in unique_teams:
            sum_Es = 0
            sum_E_for_d2 = 0

            r = self.ratings[team]
            rd = self.rds[team]

            team_matches = filtered_data[(filtered_data["left"] == team) | (filtered_data["right"] == team)]
            for _, m in team_matches.iterrows():
                left, right, winner_name = m["left"], m["right"], m["winner_name"]

                s = int(winner_name == team)
                rj = self.ratings[left if left != team else right]
                rdj = self.rds[left if left != team else right]
                E = self.__E(r, rj, rdj)
                sum_Es += self.__g(rdj) * (s - E)
                sum_E_for_d2 += (self.__g(rdj) ** 2) * E * (1 - E)

                # print(left, right, 'rj=', rj, 'RDj=', rdj, 'g(RDj)=', self.__g(rdj), 'E=', E)

            d2 = 1 / (self.q**2 * sum_E_for_d2)
            # print('d2=', d2)
            # print('---')

            new_ratings[team] += self.q / (1 / rd**2 + 1 / d2) * sum_Es
            new_rds[team] = np.sqrt(1 / (1 / rd**2 + 1 / d2))

        # Apply update once all are computed
        self.ratings.update(new_ratings)
        self.rds.update(new_rds)

    def _get_ratings_for_match(self, match_series: pd.Series):
        left, right = match_series["left"], match_series["right"]
        return pd.Series(
            {
                f"glicko_c{self.c}_left{self.suffix}": self.ratings[left],
                f"glicko_c{self.c}_right{self.suffix}": self.ratings[right],
                f"glicko_c{self.c}_rd_left{self.suffix}": self.rds[left],
                f"glicko_c{self.c}_rd_right{self.suffix}": self.rds[right],
                f"glicko_c{self.c}_rd_diff{self.suffix}": self.rds[left] - self.rds[right],
                f"glicko_c{self.c}_diff{self.suffix}": self.ratings[left] - self.ratings[right],
                f"glicko_c{self.c}_prob{self.suffix}": self._expected_scores(left, right)[left],
            }
        )

    def get_features(self, X: pd.DataFrame):
        return X.apply(self._get_ratings_for_match, axis=1)


class PlayerEloFeatureSet(FeatureSet):
    def __init__(
        self,
        k=32,
        agg="mean",
        tier_filter: int | None = None,
        tier_filter_mode: Literal["strict", "relaxed"] = "strict",
        **kwargs,
    ) -> None:
        self.elos = defaultdict(lambda: 1500.0)
        self.k = k
        self.agg = agg

        self.tier_filter = tier_filter
        self.tier_filter_mode = tier_filter_mode

        self.suffix = "" if tier_filter is None else f"_{tier_filter_mode}_tier{tier_filter}"
        super().__init__()

    def _team_elo(self, roster):
        if not isinstance(roster, Iterable):
            return 1500.0

        elos = [self.elos[p] for p in roster]
        if len(elos) == 0:
            return 1500.0

        if self.agg == "mean":
            return np.mean(elos)
        elif self.agg == "sum":
            return np.sum(elos)
        elif self.agg == "max":
            return np.max(elos)
        else:
            raise ValueError("Unknown aggregation")

    def _expected_scores(self, left_roster, right_roster):
        left_elo = self._team_elo(left_roster)
        right_elo = self._team_elo(right_roster)

        diff = right_elo - left_elo
        p_left = 1 / (1 + 10 ** (diff / 400))

        return p_left, left_elo, right_elo

    def update(self, new_data: pd.DataFrame):
        if self.tier_filter:
            if self.tier_filter_mode == "strict":
                filtered_data = new_data[
                    (new_data["left_tier"] == self.tier_filter) & (new_data["right_tier"] == self.tier_filter)
                ]
            elif self.tier_filter_mode == "relaxed":
                filtered_data = new_data[
                    (new_data["left_tier"] == self.tier_filter) | (new_data["right_tier"] == self.tier_filter)
                ]
            else:
                raise ValueError(f"Invalid tier filtering mode {self.filter_tier_mode}")
        else:
            filtered_data = new_data

        for _, row in filtered_data.iterrows():
            left_roster = row["left_roster"]
            right_roster = row["right_roster"]
            left_wins = row["left_wins"]

            p_left, _, _ = self._expected_scores(left_roster, right_roster)

            scores = {"left": left_wins, "right": 1.0 - left_wins}

            # team-level delta
            delta_left = self.k * (scores["left"] - p_left)
            delta_right = self.k * (scores["right"] - (1 - p_left))

            # distribute to players
            if isinstance(left_roster, Iterable):
                for p in left_roster:
                    self.elos[p] += delta_left / len(left_roster)

            if isinstance(right_roster, Iterable):
                for p in right_roster:
                    self.elos[p] += delta_right / len(right_roster)

    def _get_elos_for_match(self, row: pd.Series):
        left_roster = row["left_roster"]
        right_roster = row["right_roster"]

        p_left, left_elo, right_elo = self._expected_scores(left_roster, right_roster)

        return pd.Series(
            {
                f"player_elo_left{self.suffix}": left_elo,
                f"player_elo_right{self.suffix}": right_elo,
                f"player_elo_diff{self.suffix}": left_elo - right_elo,
                f"player_elo_prob{self.suffix}": p_left,
            }
        )

    def get_features(self, X: pd.DataFrame):
        return X.apply(self._get_elos_for_match, axis=1)
