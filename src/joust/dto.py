import datetime
import hashlib
from dataclasses import dataclass


@dataclass(frozen=True)
class Team:
    name: str


@dataclass(unsafe_hash=True)
class Contestant:
    """A team in a tournament"""

    team: Team
    seed: int
    initial_seed: int = 0
    wins: int = 0
    losses: int = 0

    def __post_init__(self):
        # Store the team's initial seed on instantiation
        self.initial_seed = self.seed


@dataclass
class Event:
    name: str
    start_date: datetime.date | None = None
    end_date: datetime.date | None = None


@dataclass
class Match:
    """Match with no possibility of tie."""

    left: Team
    right: Team
    date: datetime.date | None = None
    event: Event | None = None
    left_wins: bool | None = None
    best_of: int = 1

    @property
    def winner(self) -> Team | None:
        if self.left_wins is None:
            return None
        return self.left if self.left_wins else self.right

    @property
    def team_names(self) -> set[str]:
        return {self.left.name, self.right.name}

    @property
    def loser(self) -> Team | None:
        if self.left_wins is None:
            return None
        return self.right if self.left_wins else self.left

    @property
    def match_id(self) -> str:
        seed = f"{self.date.isoformat()}__{'-'.join(sorted([self.left.name, self.right.name]))}__{self.event.name}"
        return hashlib.md5(seed.encode()).hexdigest()[:8]

    def __post_init__(self):
        if self.event is None:
            self.event = Event("Mock Event")

    def __eq__(self, value: object) -> bool:
        if isinstance(value, Match):
            this_teams = set([self.left.name, self.right.name])
            other_teams = set([value.left.name, value.right.name])
            return this_teams == other_teams
        else:
            raise TypeError(f"Cannot compare Match with object of type {type(value)}")

    def to_dict(self):
        return {
            "match_id": self.match_id,
            "date": self.date,
            "left": self.left.name,
            "right": self.right.name,
            "left_wins": self.left_wins,
            "match_type": f"bo{self.best_of}",
            "event": self.event.name,
            "winner_name": self.winner.name,
        }


@dataclass
class MatchPrediction:
    match: Match
    left_wins: bool
    probability: float

    @property
    def left_odds(self) -> float:
        p = self.probability
        return p / (1 - p)

    @property
    def right_odds(self) -> float:
        p = 1.0 - self.probability
        return p / (1 - p)

    def min_acceptable_odds(self, margin: float = 0.1) -> dict[Team, float]:
        winner_odds = 1 / self.probability * (1 + margin)
        loser_odds = 1 / (1 - self.probability) * (1 + margin)
        return {
            "left" if self.left_wins else "right": winner_odds,
            "right" if self.left_wins else "left": loser_odds,
        }

    @property
    def predicted_winner(self):
        return self.match.left if self.left_wins else self.match.right

    @property
    def predicted_loser(self):
        return self.match.right if self.left_wins else self.match.left

    def to_dict(self):
        d = self.match.to_dict()
        d.update(
            {
                "left_wins_prob": self.probability,
                "left_wins_odds": self.left_odds,
                "right_wins_odds": self.right_odds,
            }
        )
        return d
