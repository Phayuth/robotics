import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def np_to_pd():
    data = np.array([(1, 2, 3), (4, 5, 6), (7, 8, 9)], dtype=[("a", "i4"), ("b", "i4"), ("c", "i4")])
    df3 = pd.DataFrame(data, columns=["c", "a"])
    print(f"> df3: {df3}")


def np_to_pd_nan():
    d = {"col1": [1, 2, None, 3, 5, 1], "col2": [3, 4, None, 3, 5, 1]}
    df = pd.DataFrame(data=d)
    print(f"> df: {df}")
    df.to_csv("out.csv", index=False)


def pd_from_dict():
    d = {"col1": [1, 2], "col2": [3, 4]}
    df = pd.DataFrame(data=d)
    print(f"> df: {df}")


def plot_table():
    df = pd.DataFrame({"A": [1, 2], "B": [3, 4]})
    fix, ax = plt.subplots()
    ax.axis("off")
    table = pd.plotting.table(ax, df, loc="center", cellLoc="center", colWidths=list([0.2, 0.2]))
    plt.show()


def pd_indexing():
    d = {"col1": [1, 2, None, 3, 5, 1], "col2": [3, 4, None, 3, 5, 1]}
    df = pd.DataFrame(data=d)
    print(f"> df: {df}")
    print(df["col1"])
    print(df[0:3])
    print(df.mean(axis=0))
    print(df.var(axis=0))


def pd_reindex():
    df = pd.DataFrame(aaa[:, 1], index=aaa[:, 0].astype(np.int32), columns=["cost"])
    new_index = pd.Index(range(2000))
    new_df = df.reindex(new_index)
    new_df = new_df.interpolate(method="nearest")
    nnn = pd.concat([new_df, new_df], axis=1)
    nnna = nnn.mean(axis=1)


def concat_own():
    # https://datatofish.com/concatenate-values-python/
    data = {"day": [1, 2, 3, 4, 5], "month": ["Jun", "Jul", "Aug", "Sep", "Oct"], "year": [2016, 2017, 2018, 2019, 2020]}
    df = pd.DataFrame(data)
    df["full_date"] = df["day"].map(str) + "-" + df["month"].map(str) + "-" + df["year"].map(str)
    print(df)


def concat_different():
    data1 = {
        "day": [1, 2, 3, 4, 5],
        "month": ["Jun", "Jul", "Aug", "Sep", "Oct"],
        "year": [2016, 2017, 2018, 2019, 2020],
    }

    df1 = pd.DataFrame(data1)

    data2 = {
        "unemployment_rate": [5.5, 5, 5.2, 5.1, 4.9],
        "interest_rate": [1.75, 1.5, 1.25, 1.5, 2],
    }

    df2 = pd.DataFrame(data2)

    combined_values = df1["day"].map(str) + "-" + df1["month"].map(str) + "-" + df1["year"].map(str) + ": " + "Unemployment: " + df2["unemployment_rate"].map(str) + "; " + "Interest: " + df2["interest_rate"].map(str)
    print(combined_values)


concat_different()
