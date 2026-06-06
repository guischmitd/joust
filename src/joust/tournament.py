import random
from abc import ABC, abstractmethod
from collections import defaultdict
from typing import Literal

from joust.dto import Contestant, Match, Team
from joust.predictor import MatchPrediction, MatchPredictor, RandomMatchPredictor

Registry = dict[Team, Contestant]


class MatchupStrategy(ABC):
    """Base class for matching teams in a round."""

    @abstractmethod
    def get_matchups(
        self, contestants: list[Contestant], match_history: list[Match], registry: Registry
    ) -> list[Match]:
        raise NotImplementedError()

    def update_seeds(self, contestants: list[Contestant], match_history: list[Match], registry: Registry) -> None:
        pass


class PureSeedMatchup(MatchupStrategy):
    """Common matchup strategy for bracket style play (playoffs)"""

    def get_matchups(
        self, contestants: list[Contestant], match_history: list[Match], registry: Registry
    ) -> list[Match]:
        assert len(contestants) > 0 and len(contestants) % 2 == 0, f"Need even >0 contestants, got {len(contestants)}"
        ordered = sorted(contestants, key=lambda c: c.seed)
        half = len(ordered) // 2
        return [Match(a.team, b.team) for a, b in zip(ordered[:half], reversed(ordered[half:]))]


class Tournament:
    def __init__(
        self,
        contestants: list[Contestant],
        round_strategies: list[MatchupStrategy],
        match_predictor: MatchPredictor | None = None,
        wins_to_advance: int = 3,
        losses_to_drop: int = 3,
        prediction_mode: Literal["sampled", "exact"] = "exact",
        update_after_round: bool = False,
    ) -> None:
        self.contestants = contestants
        self.registry: Registry = {c.team: c for c in contestants}
        self.round_strategies = round_strategies
        self.match_predictor = match_predictor or RandomMatchPredictor()
        self.current_round = 0
        self.match_history: list[Match] = []
        self.prediction_mode = prediction_mode
        self.update_after_round = update_after_round

        assert wins_to_advance > 0, f"wins_to_advance must be strictly positive (received {wins_to_advance})"
        assert losses_to_drop > 0, f"losses_to_drop must be strictly positive (received {losses_to_drop})"

        self.wins_to_advance = wins_to_advance
        self.losses_to_drop = losses_to_drop

    @property
    def active_contestants(self) -> list[Contestant]:
        return [c for c in self.contestants if c.wins < self.wins_to_advance and c.losses < self.losses_to_drop]

    @property
    def advanced(self) -> list[Contestant]:
        return [c for c in self.contestants if c.wins >= self.wins_to_advance]

    @property
    def eliminated(self) -> list[Contestant]:
        return [c for c in self.contestants if c.losses >= self.losses_to_drop]

    def _match_outcome_from_pred(self, pred: MatchPrediction) -> bool:
        if self.prediction_mode == "exact":
            return pred.left_wins
        elif self.prediction_mode == "sampled":
            roll = random.random()
            return roll < pred.probability
        else:
            raise ValueError(f"Invalid prediction_mode {self.prediction_mode}")

    def tick(self) -> list[MatchPrediction]:
        strategy = self.round_strategies[self.current_round]
        strategy.update_seeds(self.contestants, self.match_history, self.registry)
        matches = strategy.get_matchups(self.active_contestants, self.match_history, self.registry)
        predictions = [self.match_predictor.predict_winner(m) for m in matches]

        tick_matches = []
        for pred in predictions:
            pred.match.left_wins = self._match_outcome_from_pred(pred)
            self.registry[pred.match.winner].wins += 1
            self.registry[pred.match.loser].losses += 1
            tick_matches.append(pred.match)

        self.match_history.extend(tick_matches)

        self.current_round += 1

        if self.update_after_round and self.match_predictor.updateable:
            self.match_predictor.update(tick_matches)

        return predictions

    def run_to_completion(self) -> "Tournament":
        while self.active_contestants:
            self.tick()
        return self

    @property
    def pickem_outcomes(self):
        outcomes = defaultdict(list)
        for c in self.standings:
            if c.wins == 3 and c.losses == 0:
                outcomes["3:0"].append(c.team.name)
            elif c.wins == 3 and c.losses >= 1:
                outcomes["3:1|2"].append(c.team.name)
            elif c.wins == 0 and c.losses == 3:
                outcomes["0:3"].append(c.team.name)

        sorted_outcomes = {}
        for k, v in outcomes.items():
            sorted_outcomes[k] = sorted(v)

        return sorted_outcomes

    @property
    def standings(self) -> list[Contestant]:
        return sorted(self.contestants, key=lambda c: c.seed)
