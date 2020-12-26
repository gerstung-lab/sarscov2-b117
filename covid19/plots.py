from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _sort_values(model, var, j=(slice(None),), reversed=False):
    if reversed:
        idx = np.argsort(model.mean(var)[j].reshape(-1))
    else:
        idx = np.argsort(model.mean(var)[j].reshape(-1))[::-1]
    return idx


def plot_map(ax, model, var, j=(slice(None),), top=50, reversed=False):
    idx = _sort_values(m0, var, j, reversed)
    y = model.mean(var)[j].reshape(-1)

    LAD.plot(color="w", edgecolor="k", ax=ax, linewidth=0.2)

    sub_frame = LAD.iloc[idx[:top]]
    sub_frame["data"] = y[idx[:top]]
    # print(sub_frame)

    sub_frame.plot(ax=ax, column="data", cmap="Blues", linewidth=1, legend=True)
    for idx, row in sub_frame.iterrows():
        coords = row["geometry"].representative_point().coords[:][0]
        ax.annotate(
            text=str(row["lad19nm"]),
            xy=coords,
            horizontalalignment="center",
        )

    ax.axis("off")


def plot_scalar(ax, model, var, j=(slice(None),), top=50, reversed=False):
    idx = _sort_values(model, var, j, reversed)
    ci = model.ci(var)[(slice(None), *j)].reshape(2, -1)
    y = model.mean(var)[j].reshape(-1)

    yticklabels = LAD.loc[idx[:top].tolist(), "lad19nm"].tolist()
    yticklabels = [f"{label} ({i})" for label, i in zip(yticklabels, idx)]

    ax.errorbar(y[idx][:top], np.arange(top)[::-1], xerr=ci[:, idx][:, :top], fmt="o")
    _ = ax.set(yticks=np.arange(top)[::-1], yticklabels=yticklabels)
    ax.margins(0.02)
    ax.yaxis.grid(True)
    ax.xaxis.grid(True)


def plot_area(
    cases: pd.DataFrame,
    lad: pd.DataFrame = None,
    months: List[int] = [9, 10],
    areaCodes: List[str] = ["E07000237"],
    cum=False,
    legend=False,
    log=False,
):
    idx = cases.index.month.isin(months)

    for areaCode in areaCodes:
        if lad is not None:
            label = str(lad.loc[lad.lad19cd == areaCode]["lad19nm"])
        else:
            label = areaCode
        plt.plot(cases.index.values[idx], cases[areaCode][idx], "o-", label=label)
        ax = plt.gca()
        ax.set_ylabel("New Cases")
        _ = plt.setp(ax.get_xticklabels(), rotation=45, ha="right")

        if cum:
            ax2 = ax.twinx()
            ax2.plot(
                cases.index.values[idx], cases[areaCode][idx].cumsum(), ".-", color="C2"
            )
            ax2.set_ylabel("Cumulative")

    if legend:
        ax.legend()
    return ax, cases.index.values[idx]
