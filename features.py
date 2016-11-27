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


def ts_to_features(prefix, tsvar, features, cache):
    if features is True:
        features = FEATURES.keys()

    grouped = None
    for f in features:
        k = prefix + FEATURE_SEP + f

        if cache and k in cache:
            yield k, cache[k]
        else:
            if grouped is None:
                grouped = tsvar.groupby(level=0)
            res = FEATURES[f](grouped)
            cache[k] = res
            yield k, res


def df_to_features(orig, feature_mapping, cache=None):
    df = dict()
    for k in feature_mapping:
        df.update(dict(ts_to_features(k, orig[k], feature_mapping[k], cache)))

    return pd.DataFrame(df)
