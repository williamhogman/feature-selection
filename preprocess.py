import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

import metafeatures
import features

DATA_DIRECTORY = "/media/veracrypt2/data"

DIAG_CODES = [
    "D611",
    "D642",
    "D695",
    "E032",
    "E064",
    "E160",
    "E273",
    "G211",
    "G240",
    "G251",
    "G444",
    "G620",
    "I427",
    "I952",
    "L270",
    "L271",
    "M804",
    "M814",
    "N141",
    "O355",
    "R502",
    "T599",
    "T782",
    "T783",
    "T784",
    "T789",
    "T801",
    "T802",
    "T808",
    "T809",
    "T886",
    "T887",
]

path_for = "{0}/{1}-90-raw-measurements.csv".format


def read_frame(diag_code):
    return pd.read_csv(path_for(DATA_DIRECTORY, diag_code))


def get_data_for(diag_code):
    df = read_frame(diag_code)
    xs = df.pivot_table(index=["patientnr", "time"],
                        columns="code",
                        values="value")
    ys = df.groupby("patientnr")["ADE"].mean()
    return xs, ys

def accuracy(y_hat, y):
    corr = (y_test == y_hat).sum()
    total = y_test.count()
    return corr / total

def build_base_model(xs, ys):
    clf = RandomForestClassifer(n_estimators=10)
    clf.fit(xs, ys)
    return clf

def evaluate_model(clf, x_test, y_test):
    y_hat = clf.predict(x_test)
    return accuracy(y_test, y_hat)

def build_and_evaluate(features, ys):
    x_train, x_test, y_train, y_test = train_test_split(features, ys)
    clf = build_base_model(x_train, y_train)
    return evaluate_model(clf, x_test, y_test)


def compute_meta_features(df):
    return metafeatures.compute_for(df, df.columns.values)

if __name__ == "__main__":
    basedata, ys = get_data_for("T887")
    metafeatures = compute_meta_features(basedata)
    cache = dict()
    features = features.df_to_features(basedata, {k: True for k in basedata.columns.values}, cache)
    res = build_and_evaluate(features, ys)
    print(res)
