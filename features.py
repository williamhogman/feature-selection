import itertools
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

FEATURE_COMBINATIONS = list(itertools.chain.from_iterable(
    itertools.combinations(FEATURES.keys(), i)
    for i in range(1, len(FEATURES) + 1)
))

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


def df_to_all_reprs(orig, ts_under_test, cache=None):
    features = {k: True for k in orig.columns.values if not k == ts_under_test}
    for rep in FEATURE_COMBINATIONS:
        features[ts_under_test] = rep
        yield df_to_features(orig, features, cache)
