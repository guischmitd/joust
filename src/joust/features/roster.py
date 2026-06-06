from collections import defaultdict, deque

import numpy as np
import pandas as pd

from joust.features import FeatureSet


class RosterStabilityFeatureSet(FeatureSet):
    def __init__(self, window: int = 20, alpha: float = 0.3):
        self.state = defaultdict(
            lambda: {
                "last_roster": None,
                "streak": 0,
                "last_date": None,
                "ewm_overlap": 0.0,
                "changes_window": deque(maxlen=window),
                "overlap_window": deque(maxlen=window),
            }
        )
        self.window = window
        self.alpha = alpha
        super().__init__()

    # -------- utils --------
    def _is_missing(self, r):
        if r is None:
            return True
        try:
            if pd.isna(r):
                return True
        except Exception:
            pass
        return False

    def _normalize(self, roster):
        if self._is_missing(roster):
            return None
        # assume iterable of player ids
        try:
            return tuple(sorted(roster))
        except Exception:
            return None

    def _overlap(self, a, b):
        if a is None or b is None:
            return 0.0
        sa, sb = set(a), set(b)
        if not sa and not sb:
            return 0.0
        return len(sa & sb) / len(sa | sb)

    def _player_retention(self, roster, prev):
        if roster is None or prev is None or len(prev) == 0:
            return 0.0
        return len(set(roster) & set(prev)) / len(prev)

    # -------- state update --------
    def update(self, new_data: pd.DataFrame):
        for _, row in new_data.iterrows():
            for side in ["left", "right"]:
                team = row[side]
                roster = self._normalize(row.get(f"{side}_roster"))
                date = row.get("date")

                s = self.state[team]
                prev = s["last_roster"]

                ov = self._overlap(roster, prev)
                retention = self._player_retention(roster, prev)

                same = prev is not None and roster is not None and roster == prev

                # streak (count only valid comparisons)
                if same:
                    s["streak"] += 1
                else:
                    s["streak"] = 1 if roster is not None else 0

                changed = 0 if same else 1

                # rolling windows
                s["changes_window"].append(changed)
                s["overlap_window"].append(ov)

                # EWM (use retention as slightly stronger signal)
                s["ewm_overlap"] = self.alpha * retention + (1 - self.alpha) * s["ewm_overlap"]

                # update state
                s["last_roster"] = roster
                s["last_date"] = date

    # -------- feature read --------
    def _team_feats(self, team):
        s = self.state.get(team, None)

        if s is None:
            return {
                "streak": 0,
                "avg_overlap": 0.0,
                "changes": 0,
                "ewm_overlap": 0.0,
            }

        return {
            "streak": s["streak"],
            "avg_overlap": np.mean(s["overlap_window"]) if s["overlap_window"] else 0.0,
            "changes": sum(s["changes_window"]),
            "ewm_overlap": s["ewm_overlap"],
        }

    def _get_feats_for_match(self, row):
        lf = self._team_feats(row["left"])
        rf = self._team_feats(row["right"])

        return pd.Series(
            {
                "stability_streak_diff": lf["streak"] - rf["streak"],
                "stability_overlap_diff": lf["avg_overlap"] - rf["avg_overlap"],
                "stability_changes_diff": lf["changes"] - rf["changes"],
                "stability_ewm_diff": lf["ewm_overlap"] - rf["ewm_overlap"],
            }
        )

    def get_features(self, X: pd.DataFrame):
        return X.apply(self._get_feats_for_match, axis=1)
