import pandas as pd
from tqdm import tqdm


def process(df, feature_sets, freq="D", verbose=False):
    processed = []
    for date in tqdm(sorted(df["date"].unique()), disable=not verbose):
        daily_matches = df[df["date"] == date].copy()

        feats = []
        for fs in feature_sets:
            feats.append(fs.get_features(daily_matches))
            fs.update(daily_matches)

        feats = pd.concat(feats, axis=1)
        processed.append(feats)

    processed = pd.concat(processed).add_suffix(f"__{freq}")
    return processed
