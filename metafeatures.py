import pandas as pd


def variance(ts):
    return ts.var()

META_FEATURES = {
    "variance": variance
}


def extract_meta_features(timeseries):
    return {k: META_FEATURES[k](timeseries) for k in META_FEATURES}


def seq_of_timeseries_variable(df, cols):
    for col in cols:
        d = extract_meta_features(df[col])
        d["name"] = col
        yield d


def compute_for(df, cols):
    return pd.DataFrame(seq_of_timeseries_variable(df, cols))
