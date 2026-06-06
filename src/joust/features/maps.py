import warnings
from collections import defaultdict, deque

import numpy as np
import pandas as pd

from joust.features import FeatureSet

UNIQUE_MAPS = ["ovp", "nuke", "anc", "inf", "vtg", "anb", "mrg", "d2", "trn", "cch", "cbl"]
_MAP_INDICES = {m: i for i, m in enumerate(UNIQUE_MAPS)}

np.true_divide


class PlayerMapFeatures(FeatureSet):
    def __init__(self, map_window: int = 30):
        warnings.filterwarnings("ignore", "invalid value encountered in divide")
        self.map_window = map_window
        self.player_history = defaultdict(lambda: deque(maxlen=self.map_window))
        self.smoothing_alpha = 2.0
        self.smoothing_beta = 2.0

    def _get_feats_for_match(self, m):

        left_player_played = np.zeros((5, len(UNIQUE_MAPS)))
        left_player_won = np.zeros((5, len(UNIQUE_MAPS)))
        right_player_played = np.zeros((5, len(UNIQUE_MAPS)))
        right_player_won = np.zeros((5, len(UNIQUE_MAPS)))

        # Compute per roster played and won matrices
        for i, player in enumerate(m["left_roster"]):
            for map_name, won in self.player_history[player]:
                left_player_played[i, _MAP_INDICES[map_name]] += 1
                left_player_won[i, _MAP_INDICES[map_name]] += won

        for i, player in enumerate(m["right_roster"]):
            for map_name, won in self.player_history[player]:
                right_player_played[i, _MAP_INDICES[map_name]] += 1
                right_player_won[i, _MAP_INDICES[map_name]] += won

        # Compute features at player level
        left_player_wr = (left_player_won + self.smoothing_alpha) / (
            left_player_played + self.smoothing_alpha + self.smoothing_beta
        )

        right_player_wr = (right_player_won + self.smoothing_alpha) / (
            right_player_played + self.smoothing_alpha + self.smoothing_beta
        )

        # roster-level map strength
        left_wr = left_player_wr.mean(axis=0)
        right_wr = right_player_wr.mean(axis=0)

        # roster-level experience
        left_exp = np.log1p(left_player_played).mean(axis=0)
        right_exp = np.log1p(right_player_played).mean(axis=0)

        diff_wr = left_wr - right_wr
        diff_exp = left_exp - right_exp

        out = {}

        out.update(
            {f"map_wr_diff_{map_name}_{self.map_window}": value for map_name, value in zip(UNIQUE_MAPS, diff_wr)}
        )

        out.update(
            {f"map_exp_diff_{map_name}_{self.map_window}": value for map_name, value in zip(UNIQUE_MAPS, diff_exp)}
        )

        return pd.Series(out)

    def update(self, new_data: pd.DataFrame):
        for _, row in new_data.iterrows():
            for left_wins, map_name in zip(row["left_wins_per_map"], row["maps"]):
                winner_roster = row["left_roster"] if left_wins else row["right_roster"]
                loser_roster = row["right_roster"] if left_wins else row["left_roster"]

                for player in loser_roster:
                    self.player_history[player].append((map_name, 0))
                for player in winner_roster:
                    self.player_history[player].append((map_name, 1))

    def get_features(self, X: pd.DataFrame):
        result = X.apply(self._get_feats_for_match, axis=1)
        return result
