import numpy as np


def add_pdays_features(dataframe):
    data = dataframe.copy()
    data["pdays_no_prev_contact"] = (data["pdays"] == -1).astype(int)
    data["pdays"] = data["pdays"].replace(-1, np.nan)
    return data
