import pandas as pd

DATA_DIRECTORY = "/media/veracrypt1/data"

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
    return read_frame(diag_code).pivot_table(index=["patientnr", "time"], columns="code", values="value")


if __name__ == "__main__":
    print(get_data_for("T887").head())
