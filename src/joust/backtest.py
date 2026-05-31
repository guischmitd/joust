import datetime
from collections import defaultdict

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, classification_report, log_loss, roc_auc_score
from tqdm import tqdm

from joust.dto import Match
from joust.features import process
from joust.features import FeatureSet
from joust.predictor import MatchPredictor
from joust.utils import dataframe_from_matches


def row_count(y_true, y_prob):
    return len(y_true)


def expected_calibration_error(y_true, y_prob, n_bins=20, **kwargs):
    y_true = np.asarray(y_true).ravel()
    y_prob = np.asarray(y_prob).ravel()

    bins = np.linspace(0.0, 1.0, n_bins + 1)
    bin_ids = np.digitize(y_prob, bins) - 1

    ece = 0.0
    N = len(y_true)

    for b in range(n_bins):
        mask = bin_ids == b
        if not np.any(mask):
            continue

        acc = y_true[mask].mean()
        conf = y_prob[mask].mean()
        ece += (mask.sum() / N) * abs(acc - conf)

    return ece


def binary_log_loss(y_true, y_prob):
    return log_loss(y_true.astype(int), y_prob, labels=[0, 1])


def binary_brier_score_loss(y_true, y_prob):
    return brier_score_loss(y_true.astype(int), y_prob, labels=[0, 1])


DEFAULT_METRICS = [expected_calibration_error, binary_brier_score_loss, binary_log_loss, roc_auc_score, row_count]


def evaluate(matches: pd.DataFrame, probs: np.ndarray, metrics=DEFAULT_METRICS, report=True, name="evaluation"):
    if report:
        print(classification_report(matches["left_wins"], probs.round()))

    return pd.Series({m.__name__: m(matches["left_wins"], probs) for m in metrics}, name=name)


def test_big_events(df, backend, min_date="2025-06-01", test_name=None, report=False):
    event_mask = df["is_big_event"].astype(bool)
    # print(event_mask.sum(), 'rows in big events')

    test_name = test_name or "Big events"

    event_start = max(df.loc[event_mask, "date"].min().date().isoformat(), min_date)
    event_end = df.loc[event_mask, "date"].max()

    all_probs = []
    for date in tqdm(pd.date_range(event_start, event_end), desc=f"Running CV on {test_name}"):
        history = df[df["date"] < date]
        matches = df[(df["date"] == date) & event_mask].copy()

        if len(matches):
            backend.fit(history, history["left_wins"])
            p = backend.predict_proba(matches)[:, -1]

            all_probs.append(pd.Series(p.ravel(), index=matches.index, name="probs"))

    probs = pd.concat(all_probs)
    matches = df.loc[event_mask & df["date"].between(event_start, event_end)]
    result = evaluate(matches, probs.loc[matches.index], report=report, name=test_name)

    return result, probs.loc[matches.index]


class Backtest:
    def __init__(
        self,
        predictors: dict[str, MatchPredictor],
        history: pd.DataFrame | list[Match],
        test_mask: pd.Series | None = None,
        start_date: datetime.date | None = None,
        end_date: datetime.date | None = None,
        features: list[FeatureSet] | None = None,
    ) -> None:
        """
        Backtest a set of predictors over a span of days or a specific set of matches.
        """

        span_provided = all([d is not None for d in (start_date, end_date)])
        mask_provided = test_mask is not None
        if span_provided and mask_provided:
            raise ValueError("Provide either `test_mask` or (`start_date`, `end_date`), not both.")
        elif span_provided or mask_provided:
            raise ValueError("Either `test_mask` or (`start_date`, `end_date`) must be provided.")

        if isinstance(history, list):
            self.history_df = dataframe_from_matches(history)
        else:
            self.history_df = history

        if span_provided:
            self.test_mask = self.history_df["date"].between(start_date, end_date, inclusive="both")
            self.dates = pd.date_range(start_date, end_date, inclusive="both")
        else:
            self.test_mask = test_mask
            self.dates = self.history_df.loc[test_mask]["date"].unique().sort().tolist()

        self.feature_sets = features
        self.predictors = predictors
        self.evaluation_results = []

    def _compute_features(self):
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

    def _evaluate_default_metrics(self, matches, probs, name='evaluation'):
        return evaluate(matches, probs, report=False, name=name)
    
    def _evaluate_upset_metrics(self):
        pass