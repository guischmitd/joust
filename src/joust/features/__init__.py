from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass, field

import pandas as pd
from tqdm import tqdm


class FeatureSet(ABC):
    def __init__(self) -> None:
        self.history = None

    def set_fequency(self, freq):
        self.freq = freq

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


def process(df, feature_sets, freq="D", verbose=False):  # noqa: C901
    # assert df['match_datetime'].is_monotonic_increasing, 'Matches must be provided in monotonic ascending order'
    # TODO assertion doesnt work on map level
    date_range = pd.date_range(
        df["date"].min(),
        df["date"].max(),
        freq=freq,
        inclusive="both",
    ).tolist()

    if verbose:
        print(f"Processing matches from {df['match_datetime'].min()} to {df['match_datetime'].max()}")
        print(f"Generated date range from {date_range[0]} to {date_range[-1]}")

    # Make sure the range covers all rows in the dataset
    if date_range[0] >= df["date"].min():
        if verbose:
            print("Padding left of date range")
        date_range = ["1962-02-25", *date_range]

    if date_range[-1] < df["date"].max():
        if verbose:
            print("Padding right of date range")
        date_range.append("2126-06-30")

    processed = []

    iterator = zip(date_range[:-1], date_range[1:])

    for start_date, end_date in tqdm(iterator, total=len(date_range) - 1, disable=not verbose):
        period_matches = df[(df["date"] > start_date) & (df["date"] <= end_date)]

        if len(period_matches):
            feats = pd.concat(
                [fs.get_features(period_matches) for fs in feature_sets],
                axis=1,
            )
        else:
            feats = []

        for fs in feature_sets:
            fs.update(period_matches)

        if len(feats):
            processed.append(feats)

    if not processed:
        return pd.DataFrame()

    return pd.concat(processed).add_suffix(f"__{freq}")


@dataclass
class FeatureGroup:
    name: str
    entity: str  # "team_name", "player_name", etc.
    time_key: str  # "date" or "match_datetime"
    compute: Callable  # facts → pd.DataFrame
    feature_cols: list[str] = field(default_factory=list)

    def build(self, facts: dict[str, pd.DataFrame]) -> pd.DataFrame:
        df = self.compute(facts)
        assert self.entity in df.columns
        assert self.time_key in df.columns
        self.feature_cols = [c for c in df.columns if c not in (self.entity, self.time_key, "map_id")]
        return df


def build_training_set(
    matches: pd.DataFrame,
    feature_groups: list[FeatureGroup],
    facts: dict[str, pd.DataFrame],
    sides: tuple[str, str] = ("left", "right"),
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
