from collections import defaultdict
from copy import deepcopy

import pandas as pd
from tqdm import tqdm

from joust.dto import Match
from joust.features import process
from joust.features.base import FeatureSet
from joust.predictor import MatchPredictor, RandomMatchPredictor
from joust.tournament import Tournament
from joust.utils import dataframe_to_matches


class Simulator:
    def __init__(
        self,
        tournament: Tournament,
        history: pd.DataFrame | list[Match],
        features: list[FeatureSet],
        match_predictor: MatchPredictor | None = None,
        n_sims: int = 100,
    ) -> None:

        self.n_sims = n_sims
        self.tournament = tournament
        if isinstance(history, list):
            self.history = dataframe_to_matches(history)
        else:
            self.history = history
        self.feature_sets = features

        self.match_predictor = match_predictor or RandomMatchPredictor()

        self.__compute_features()

        self.sim_outcomes = []

    def __compute_features(self):
        fsets_per_freq = defaultdict(list)

        for fs in self.feature_sets:
            fsets_per_freq[fs.freq].append(fs)

        feats = pd.concat(
            [process(df=self.history, feature_sets=fsets, freq=freq) for freq, fsets in fsets_per_freq.items()],
            axis=1,
        )
        self.history = self.history.join(feats)

    def run(self):
        for _ in tqdm(range(self.n_sims)):
            t = deepcopy(self.tournament)
            t.run_to_completion()
            self.sim_outcomes.append(t)
