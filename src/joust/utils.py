import pandas as pd

from joust.dto import Event, Match, MatchPrediction, Team


def nullable_bool(v):
    return bool(v) if v is not None else None


def match_from_dict(data) -> Match:
    return Match(
        date=data["date"],
        left=Team(data["left"]),
        right=Team(data["right"]),
        event=Event(data.get("event", "Mock event")),
        left_wins=nullable_bool(data.get("left_wins")),
        best_of=int(data["match_type"].replace("bo", "")),
    )


def matches_from_dataframe(df) -> list[Match]:
    return [match_from_dict(row) for _, row in df.iterrows()]


def dataframe_from_matches(matches: list[Match | MatchPrediction]):
    return pd.concat([pd.Series(m.to_dict()) for m in matches], axis=1).T
