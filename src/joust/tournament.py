from abc import ABC, abstractmethod
from joust.dto import Team, Match
from joust.predictor import MatchPredictor

class MatchupStrategy(ABC):
    """
    Base class for the strategy used for matching teams of a single swiss group.
    Strategies can be composed for each round to create custom swiss matchup logic
    """
    def __init__(self) -> None:
        pass

    @abstractmethod
    def get_matchups(self, teams: list[Team], history: list[Match]) -> list[Match]:
        raise NotImplementedError()

class PureSeedMatchup(MatchupStrategy):
    def __init__(self) -> None:
        super().__init__()

    def get_matchups(self, teams: list[Team], history: list[Match]) -> list[Match]:
        assert len(teams) > 0 and len(teams) % 2 == 0, ValueError(f"Number of teams must be even and greater than 0. Received: {len(teams)}")

        sorted_teams = sorted(teams, key=lambda x: x.seed)[::-1]
        half_length = len(sorted_teams) // 2
        matches = [
            Match(a, b)
            for a, b in zip(sorted_teams[:half_length], sorted_teams[-1:-half_length-1:-1])
        ]

        return matches

class Tournament:
    def __init__(self,
                 teams: list[Team],
                 round_strategies: list[MatchupStrategy],
                 match_predictor: MatchPredictor) -> None:
        self.teams = teams
        self.round_strategies = round_strategies
        self.match_predictor = match_predictor
        self.current_round = 0

        self._match_history = []

    def tick(self):
        matches = self._generate_matches()
        predictions = self._predict_matches(matches)


    def _predict_matches(self, matches: list[Match]):
        return [
            self.match_predictor.predict_winner(match)
            for match in matches
        ]

    def _generate_matches(self):
        this_round = self.round_strategies[self.current_round]
        matchups = this_round.get_matchups(self.teams, self._match_history)
        self.current_round += 1
        return matchups