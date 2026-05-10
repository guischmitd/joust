from collections import defaultdict

from joust.dto import Contestant, Match

VALVE_ROUNDS_4_5_PRIORITY_MATCHUPS = [
    [(0, 5), (1, 4), (2, 3)],  # 1v6 2v5 3v4
    [(0, 5), (1, 3), (2, 4)],  # 1v6 2v4 3v5
    [(0, 4), (1, 5), (2, 3)],  # 1v5 2v6 3v4
    [(0, 4), (1, 3), (2, 5)],  # 1v5 2v4 3v6
    [(0, 3), (1, 5), (2, 4)],  # 1v4 2v6 3v5
    [(0, 3), (1, 4), (2, 5)],  # 1v4 2v5 3v6
    [(0, 5), (1, 2), (3, 4)],  # 1v6 2v3 4v5
    [(0, 4), (1, 2), (3, 5)],  # 1v5 2v3 4v6
    [(0, 2), (1, 5), (3, 4)],  # 1v3 2v6 4v5
    [(0, 2), (1, 4), (3, 5)],  # 1v3 2v5 4v6
    [(0, 3), (1, 2), (4, 5)],  # 1v4 2v3 5v6
    [(0, 2), (1, 3), (4, 5)],  # 1v3 2v4 5v6
    [(0, 1), (2, 5), (3, 4)],  # 1v2 3v6 4v5
    [(0, 1), (2, 4), (3, 5)],  # 1v2 3v5 4v6
    [(0, 1), (2, 3), (4, 5)],  # 1v2 3v4 5v6
]


def buchholz_difficulty_score(contestant: Contestant, history: list[Match], contestants: list[Contestant]) -> int:
    """Buchholz: sum of (wins - losses) of all opponents faced."""
    team_to_contestant = {c.team: c for c in contestants}
    score = 0
    for m in history:
        if m.left_wins is None:
            continue
        if m.left is contestant.team:
            opp_team = m.right
        elif m.right is contestant.team:
            opp_team = m.left
        else:
            continue
        opp = team_to_contestant[opp_team]
        score += opp.wins - opp.losses
    return score


def recalculate_buchholz_seeds(contestants: list[Contestant], history: list[Match]) -> None:
    """Re-seed in place by: W-L record (desc), Buchholz (desc), initial seed (asc)."""
    contestants.sort(
        key=lambda c: (
            -(c.wins - c.losses),
            -buchholz_difficulty_score(c, history, contestants),
            c.initial_seed,
        )
    )
    for i, c in enumerate(contestants):
        c.seed = i + 1


class ValveBuchholzMatchup:
    def __init__(self, round_number: int) -> None:
        if 1 <= round_number <= 5:
            self.round = round_number
        else:
            raise ValueError(f"round_number must be >= 1 and <= 5. Received {round_number}")

    def update_seeds(self, contestants: list[Contestant], match_history: list[Match]) -> None:
        self.match_history = match_history
        recalculate_buchholz_seeds(contestants, match_history)

    def _get_groups(self, contestants: list[Contestant]) -> dict[str, list[Contestant]]:
        groups: dict[str, list[Contestant]] = defaultdict(list)
        for c in contestants:
            groups[f"{c.wins}-{c.losses}"].append(c)
        return groups

    def _teams_met_before(self, left: Contestant, right: Contestant) -> bool:
        return Match(left.team, right.team) in self.match_history

    def __recursive_matchup(self, sorted_contestants: list[Contestant]) -> list[Match] | None:
        if not sorted_contestants:
            return []

        left = sorted_contestants[0]
        for right in reversed(sorted_contestants[1:]):
            if not self._teams_met_before(left, right):
                remaining = [c for c in sorted_contestants if c not in (left, right)]
                result = self.__recursive_matchup(remaining)
                if result is not None:
                    return [Match(left.team, right.team), *result]

        return None

    def __round_1_matchups(self, sorted_contestants: list[Contestant]) -> list[Match]:
        n = len(sorted_contestants)
        assert n == 16, f"First round of ValveBuchholz must have exactly 16 teams. Received {n}"
        first_seeds = sorted_contestants[: n // 2]
        last_seeds = sorted_contestants[n // 2 :]
        return [Match(left.team, right.team, best_of=1) for left, right in zip(first_seeds, last_seeds)]

    def __round_2_matchups(self, sorted_contestants: list[Contestant]) -> list[Match]:
        n = len(sorted_contestants)
        halfpoint = n // 2
        return [
            Match(left.team, right.team)
            for left, right in zip(sorted_contestants[:halfpoint], reversed(sorted_contestants[halfpoint:]))
        ]

    def __round_3_matchups(self, sorted_contestants: list[Contestant]) -> list[Match]:
        matchups = self.__recursive_matchup(sorted_contestants)
        if matchups is None:
            raise RuntimeError("No valid non-rematch pairing found")
        return matchups

    def __round_4_5_matchups(self, sorted_contestants: list[Contestant]) -> list[Match]:
        n = len(sorted_contestants)
        if n != 6:
            raise ValueError(f"Valve priority pairing only supported for 6-team groups. Got {n} teams.")

        for pattern in VALVE_ROUNDS_4_5_PRIORITY_MATCHUPS:
            matchups = []
            valid = True
            for left_idx, right_idx in pattern:
                left = sorted_contestants[left_idx]
                right = sorted_contestants[right_idx]
                if self._teams_met_before(left, right):
                    valid = False
                    break
                matchups.append(Match(left.team, right.team))
            if valid:
                return matchups

        raise RuntimeError("No valid non-rematch pairing found for round 4/5.")

    def _get_matchups_for_group(self, contestants: list[Contestant]) -> list[Match]:
        sorted_contestants = sorted(contestants, key=lambda c: c.seed)
        if self.round == 1:
            return self.__round_1_matchups(sorted_contestants)

        if self.round == 2:
            return self.__round_2_matchups(sorted_contestants)

        if self.round == 3:
            return self.__round_3_matchups(sorted_contestants)

        if self.round in (4, 5):
            return self.__round_4_5_matchups(sorted_contestants)

    def get_matchups(self, contestants: list[Contestant], match_history: list[Match]) -> list[Match]:
        self.match_history = match_history
        groups = self._get_groups(contestants)
        matchups = []
        for group in groups.values():
            matchups.extend(self._get_matchups_for_group(group))
        return matchups
