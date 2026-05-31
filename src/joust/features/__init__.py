from dataclasses import dataclass, field
from typing import Callable

import pandas as pd
from tqdm import tqdm

from abc import ABC, abstractmethod

import pandas as pd


class FeatureSet(ABC):
    def __init__(self) -> None:
        self.history = None

    def _update_history(self, new_data: pd.DataFrame):
        if self.history is None:
            self.history = new_data
        else:
            self.history = pd.concat([self.history, new_data], axis=0)

    @abstractmethod
    def update(self, new_data: pd.DataFrame):
        """Updates the current state given a new batch of data"""
        pass

    @abstractmethod
    def get_features(self, X: pd.DataFrame):
        """Returns current values of features to all entities in X"""
        pass


def process(df, feature_sets: list[FeatureSet], freq="D", verbose=False):
    processed = []
    for date in tqdm(sorted(df["date"].unique()), disable=not verbose):
        daily_matches = df[df["date"] == date].copy()

        feats = []
        for fs in feature_sets:
            feats.append(fs.get_features(daily_matches))
            fs.update(daily_matches)

        feats = pd.concat(feats, axis=1)
        processed.append(feats)

    processed = pd.concat(processed).add_suffix(f"__{freq}")
    return processed


@dataclass
class FeatureGroup:
    name: str
    entity: str          # "team_name", "player_name", etc.
    time_key: str        # "date" or "match_datetime"
    compute: Callable    # facts → pd.DataFrame
    feature_cols: list[str] = field(default_factory=list)

    def build(self, facts: dict[str, pd.DataFrame]) -> pd.DataFrame:
        df = self.compute(facts)
        assert self.entity in df.columns
        assert self.time_key in df.columns
        self.feature_cols = [c for c in df.columns
                             if c not in (self.entity, self.time_key, "map_id")]
        return df


def build_training_set(
    matches: pd.DataFrame,
    feature_groups: list[FeatureGroup],
    facts: dict[str, pd.DataFrame],
    sides: tuple[str, str] = ("left", "right")
) -> pd.DataFrame:
    result = matches.copy()

    for group in feature_groups:
        features = group.build(facts)  # or load from cache

        for side, team_col in zip(("a", "b"), sides):
            joined = (
                matches[["match_hash", "date", team_col]]
                .rename(columns={team_col: group.entity})
                .merge(features, on=[group.entity, "date"], how="left")
                .rename(columns={c: f"{side}__{c}" for c in group.feature_cols})
                .drop(columns=[group.entity, "date"])
            )
            result = result.merge(joined, on="match_hash", how="left")

    return result

