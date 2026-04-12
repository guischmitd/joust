from dataclasses import dataclass


@dataclass(unsafe_hash=True)
class Team:
    name: str
    seed: int
    initial_seed: int = 0
    wins: int = 0
    losses: int = 0

    def __post_init__(self):
        if self.initial_seed == 0:
            self.initial_seed = self.seed


@dataclass
class Match:
    """Match with no possibility of tie."""

    left: Team
    right: Team
    left_wins: bool | None = None
    best_of: int = 1

    @property
    def winner(self) -> Team | None:
        if self.left_wins is None:
            return None
        return self.left if self.left_wins else self.right

    @property
    def loser(self) -> Team | None:
        if self.left_wins is None:
            return None
        return self.right if self.left_wins else self.left
