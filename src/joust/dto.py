from typing import Union
from dataclasses import dataclass

@dataclass
class Team:
    name: str
    seed: int
    wins: int
    losses: int

@dataclass
class Match:
    """
    Default object representing a Match with no possibility of tie
    """
    left: Team
    right: Team
    left_wins: Union[bool, None]
    
    @property
    def winner(self):
        if self.left_wins is None:
            return None
        return self.left if self.left_wins else self.right
