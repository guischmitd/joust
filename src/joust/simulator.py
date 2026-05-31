from collections import defaultdict
from copy import deepcopy

import pandas as pd
from tqdm import tqdm

from joust.dto import Match
from joust.features import process
from joust.features import FeatureSet
from joust.predictor import MatchPredictor, RandomMatchPredictor
from joust.tournament import Tournament
from joust.utils import dataframe_from_matches


class Simulator:
    def __init__(
        self,
        tournament: Tournament,
        features: list[FeatureSet] | None = None,
        history: pd.DataFrame | list[Match] | None = None,
        match_predictor: MatchPredictor | None = None,
        n_sims: int = 100,
    ) -> None:

        self.n_sims = n_sims
        self.tournament = tournament

        if isinstance(history, list):
            self.history_df = dataframe_from_matches(history)
        else:
            self.history_df = history

        self.feature_sets = features

        self.match_predictor = match_predictor or RandomMatchPredictor()

        self.__initialise_predictor()

        self.sim_outcomes = []

    def __initialise_predictor(self):
        if self.feature_sets:
            print(f"Precomputing feature sets: {[fs.__class__.__name__ for fs in self.feature_sets]}")
            fsets_per_freq = defaultdict(list)

            for fs in self.feature_sets:
                fsets_per_freq[fs.freq].append(fs)

            feats = pd.concat(
                [process(df=self.history_df, feature_sets=fsets, freq=freq) for freq, fsets in fsets_per_freq.items()],
                axis=1,
            )
            self.history_df = self.history_df.join(feats)

        print(f"Fitting {self.match_predictor.__class__.__name__}")
        self.match_predictor.fit(self.history_df)

    def run(self):
        for _ in tqdm(range(self.n_sims)):
            t = deepcopy(self.tournament)
            t.prediction_mode = "sampled"  # For bootstrapping
            t.match_predictor = self.match_predictor
            self.sim_outcomes.append(t.run_to_completion())

    def summary(self):
        if self.sim_outcomes:
            team_outcomes = defaultdict(lambda: dict(three_zero=0, zero_three=0, advanced=0, eliminated=0))
            for tournament in self.sim_outcomes:
                for contestant in tournament.standings:
                    if contestant.wins == 3 and contestant.losses == 0:
                        team_outcomes[contestant.team.name]["three_zero"] += 1
                    elif contestant.wins == 0 and contestant.losses == 3:
                        team_outcomes[contestant.team.name]["zero_three"] += 1
                    elif contestant.wins == 3:
                        team_outcomes[contestant.team.name]["advanced"] += 1
                    elif contestant.losses == 3:
                        team_outcomes[contestant.team.name]["eliminated"] += 1

            return pd.DataFrame(team_outcomes).T

        else:
            raise RuntimeError("No outcomes recorded, call Simulator.run() first.")
