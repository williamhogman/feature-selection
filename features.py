import pandas as pd


def mean(x):
    return x.mean()


def sd(x):
    return x.std()


def count(x):
    return x.count()


FEATURES = {
    "mean": mean,
    "sd": sd,
    "count": count,
}
FEATURE_SEP = "_"


def ts_to_features(prefix, tsvar, features):
    if features is True:
        features = FEATURES.keys()

    grouped = tsvar.groupby(level=0)
    return {prefix + FEATURE_SEP + f: FEATURES[f](grouped) for f in features}


def df_to_features(orig, feature_mapping):
    df = dict()
    for k in feature_mapping:
        df.update(ts_to_features(k, orig[k], feature_mapping[k]))

    return pd.DataFrame(df)
