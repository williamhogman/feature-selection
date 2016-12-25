import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import metafeatures
import features
from constants import DIAG_CODES, TRAINING_FILES

DATA_DIRECTORY = "/media/veracrypt2/data"
#DATA_DIRECTORY = "data"
DATA_DIRECTORY = "/volumes/NO NAME/data"

TEST_FILES = [x for x in DIAG_CODES if x not in TRAINING_FILES]

path_for = "{0}/{1}-90-raw-measurements.csv".format

BASE_MODEL_N_ESTIMATORS = 1
META_MODEL_N_ESTIMATORS = 1


def read_frame(diag_code):
    return pd.read_csv(path_for(DATA_DIRECTORY, diag_code))

FILL_NA_VAL = -100000

def get_data_for(diag_code):
    df = read_frame(diag_code)
    xs = df.pivot_table(index=["patientnr", "time"],
                        columns="code",
                        values="value")
    ys = df.groupby("patientnr")["ADE"].mean()
    return xs, ys

def accuracy(y_hat, y_test):
    corr = (y_test == y_hat).sum()
    total = y_hat.shape[0]
    return corr / float(total)

def build_base_model(xs, ys):
    clf = RandomForestRegressor(n_estimators=BASE_MODEL_N_ESTIMATORS, n_jobs=-1, max_features=None)
    clf.fit(xs, ys)
    return clf

def evaluate_model(clf, x_test, y_test):
    y_hat = clf.predict(x_test)
    return accuracy(y_test, y_hat)

def build_and_evaluate(features, ys):
    x_train, x_test, y_train, y_test = train_test_split(features.fillna(FILL_NA_VAL), ys)
    clf = build_base_model(x_train, y_train)
    return evaluate_model(clf, x_test, y_test)


def iter_all_tses(datasets):
    for (basedata, ys) in datasets:
        for tsvar in basedata.columns.values:
            yield basedata, ys, tsvar


def build_meta_model_ys(datasets):
    last_basedata = None
    cache = None
    i = 0
    for (basedata, ys, tsvar) in iter_all_tses(datasets):
        if cache is None or basedata is not last_basedata:
            cache = dict()
            last_basedata = basedata
        dfs = features.df_to_all_reprs(basedata, tsvar, cache)
        yield [build_and_evaluate(df, ys) for df in dfs]
        i += 1
        if i % 100 == 0:
            print("i = {0}".format(i))


def build_meta_model_xs(datasets):
    for (basedata, ys, tsvar) in iter_all_tses(datasets):
        yield metafeatures.extract_meta_features_as_arr(basedata[tsvar])


def build_meta_model(xs, ys):
    clf = RandomForestRegressor(n_estimators=META_MODEL_N_ESTIMATORS, n_jobs=-1, max_features=None)
    clf.fit(xs, ys)
    return clf

def meta_build_and_evaluate(xs, ys):
    x_train, x_test, y_train, y_test = train_test_split(xs, ys)
    clf = build_meta_model(x_train, y_train)
    y_hat = clf.predict(x_test)
    return clf, y_hat, y_train, ((y_hat - y_test) ** 2).mean()


def select_features_based_on_accuracy(accs):
    return np.array([features.FEATURE_COMBINATIONS[int(x)]
                     for x in accs.argmax(axis=1)])


def get_meta_model_xs():
    try:
        return np.load("meta-model-xs.npy")
    except:
        data = [get_data_for(x) for x in TRAINING_FILES]
        print("Building X-values for meta-model")

        xs = np.array(list(build_meta_model_xs(data)))
        np.save("meta-model-xs.npy", xs)
        return xs


def get_meta_model_ys():
    try:
        return np.load("meta-model-ys.npy")
    except:
        data = [get_data_for(x) for x in TRAINING_FILES]
        print("Building Y-values for meta-model")
        ys = np.array(list(build_meta_model_ys(data)))
        np.save("meta-model-ys.npy", ys)
        return ys


def meta_model_evaluation_run():
    xs = get_meta_model_xs()
    ys = get_meta_model_ys()
    print("Building meta model")
    clf, y_hat, y_train, meansquareerror = meta_build_and_evaluate(xs, ys)
    chosen = select_features_based_on_accuracy(y_hat)
    actual = select_features_based_on_accuracy(y_train)
    tot = 0
    for i in range(len(chosen)):
        print(chosen[i], actual[i])
        if chosen == actual:
            tot += 1
    print tot


def meta_model_train_all_run():
    xs = get_meta_model_xs()
    ys = get_meta_model_ys()
    return build_meta_model(xs, ys)


def pick_representation_for_ts(basedata, tsvar, clf):
    metas = metafeatures.extract_meta_features_as_arr(basedata[tsvar])
    return select_features_based_on_accuracy(clf.predict([metas]))[0]


def make_smart_xs(basedata, clf):
    feature_mapping = {
        tsvar: pick_representation_for_ts(basedata, tsvar, clf)
        for tsvar in basedata.columns.values
    }
    return features.df_to_features(basedata, feature_mapping, dict())


def make_naive_xs(basedata):
    feature_mapping = {k: True for k in basedata.columns.values}
    return features.df_to_features(basedata, feature_mapping, dict())


def get_experiments(clf, files):
    for (basedata, ys) in [get_data_for(x) for x in files]:
        smart_xs = make_smart_xs(basedata, clf)
        naive_xs = make_navive_xs(basedata)
        yield (naive_xs, smart_xs, ys)



def generate_comparison(naive_results, smart_results):
    return dict(naive=naive_results,
                smart=smart_results,
                delta=(naive_results - smart_results))


def run_experiments(experiments):
    for naive_xs, smart_xs, ys in experiments:
        yield generate_comparison(build_and_evaluate(naive_xs, ys),
                                  build_and_evaluate(smart_xs, ys))


def run_evaluation_procedure():
    clf = meta_model_train_all_run()
    exprs = get_experiments(clf, TEST_FILES)
    res = pd.DataFrame(list(run_experiments(exprs)))
    res.to_csv("final-results.csv")

if __name__ == "__main__":
    run_evaluation_procedure()
