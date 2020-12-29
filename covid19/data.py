# import geopandas as gpd
import pandas as pd
from collections import Counter

# from uk_covid19 import Cov19API

from covid19.config import API, Files


def get_lads(lad_path=Files.LAD):
    return pd.read_csv(lad_path, index_col=0)


def get_utla_data(utla_path=Files.UTLA_PATH):
    utla = (
        pd.read_csv(utla_path)
        .drop(["FID", "LTLA19NM"], 1)
        .rename(
            columns=lambda x: {
                "LTLA19CD": "lad19cd",
                "UTLA19CD": "utla19cd",
                "UTLA19NM": "utla19nm",
            }[x]
        )
        .assign(
            utla19cts=lambda df: df.utla19cd.apply(lambda x: Counter(df.utla19cd)[x])
        )
    )
    return utla


def get_lookup_data(reg_path=Files.REG_PATH):
    reg = (
        pd.read_csv(reg_path)
        .drop(["FID", "WD19CD", "WD19NM", "LAD19NM"], 1)
        .drop_duplicates()
        .rename(columns=lambda x: x.lower())
    )
    return reg


def get_popdata(pop_path=Files.POP_PATH):
    pop18 = (
        pd.read_csv(pop_path, index_col=0)
        .rename(
            columns=lambda x: "pop18"
            if x == "v4_0"
            else ("lad19cd" if x == "admin-geography" else x)
        )
        .loc[:, ["pop18", "lad19cd"]]
    )
    return pop18


def get_raw_cases(cases_path: str = Files.CASES_PATH):
    cases = pd.read_csv(cases_path, index_col="date")
    return cases


def get_cases(cases_path: str = Files.CASES_PATH):
    cases = (
        pd.read_csv(cases_path, index_col="date")
        .reset_index()
        .assign(date=lambda df: pd.DatetimeIndex(df.date))
        .sort_values(by="date")
        .set_index("date", drop=True)
    )
    return cases


def get_geodata(lad_path=Files.LAD_PATH):
    lad = gpd.read_file(lad_path)
    reg = get_lookup_data()
    pop = get_popdata()
    utla = get_utla_data()
    utla_codes = dict(
        zip(
            utla[utla.utla19cts > 1].utla19cd,
            utla[utla.utla19cts > 1].utla19cd.astype("category").cat.codes.values + 4,
        )
    )
    # cases = get_raw_cases()

    lad = (
        lad.merge(reg, on="lad19cd", how="left")
        .merge(pop, on="lad19cd", how="left")
        .merge(utla, on="lad19cd", how="left")
        # .merge(cases.T.reset_index(), left_on="lad19cd", right_on="index", how="left")
        .assign(lad19id=lambda df: df.lad19cd.astype("category").cat.codes.values)
        .assign(ctry19id=lambda df: df.ctry19cd.astype("category").cat.codes.values)
        .assign(
            utla19id=lambda df: df.apply(
                lambda row: utla_codes[row.utla19cd]
                if row.utla19cd in utla_codes.keys()
                else row.ctry19id,
                1,
            )
        )
        # .drop("index", 1)
        .loc[
            :,
            [
                "objectid",
                "pop18",
                "lad19cd",
                "lad19nm",
                "lad19id",
                "utla19cd",
                "utla19nm",
                "utla19id",
                # "utla19cts",
                # "lad19nmw",
                # "cty19cd",
                # "cty19nm",
                # "rgn19cd",
                # "rgn19nm",
                "ctry19cd",
                "ctry19nm",
                "ctry19id",
                "bng_e",
                "bng_n",
                "long",
                "lat",
                "st_areasha",
                "st_lengths",
                "geometry",
            ],
        ]
    )
    return lad


def extract_covariate(dataframes, covariate="newCasesByPublishDate"):
    filtered = pd.concat(
        [
            current_df.drop_duplicates()
            # .assign(date=lambda df: pd.DatetimeIndex(df.date))
            .sort_values(by="date")
            .reset_index(drop=True)
            .loc[:, ["date", "areaCode", covariate]]
            for current_df in dataframes
        ]
    ).reset_index(drop=True)

    pivot = (
        filtered.pivot_table(
            index="date", columns="areaCode", values=covariate, dropna=False
        )
        .reset_index()
        .rename_axis(None, axis=1)
        .set_index("date")
    )

    return pivot


def get_lad_data(LAD, structure=API.cases_and_deaths):
    dataframes = []
    for i, row in LAD.iterrows():
        # print(f"{row['lad19nm']}")
        api = Cov19API(filters=[f"areaCode={row['lad19cd']}"], structure=structure)
        df = api.get_dataframe()
        if df.shape == (0, 0):
            continue
        dataframes.append(df)

    return dataframes
