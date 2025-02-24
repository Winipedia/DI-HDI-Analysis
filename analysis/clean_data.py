"""
In this file I read the datasets to dfs and clean them 
or define functions to get them in a specific format for further analysis.
"""

import pandas as pd
import os
from utils.util import convert_country_col_to_iso3


DI_COL = "democracy_index"
HDI_COL = "human_development_index"
YEAR_COL = "year"
COUNTRY_COL = "country"

COL_TYPE = {
    DI_COL: float,
    HDI_COL: float,
    YEAR_COL: int,
    COUNTRY_COL: str,
}

COL_CONVERTER = {
    DI_COL: lambda x: x * 10,
    HDI_COL: lambda x: x * 100,
    YEAR_COL: lambda x: x,
    COUNTRY_COL: convert_country_col_to_iso3,
}

ALL_COLS = list(COL_TYPE.keys())



def get_democracy_index():
    
    di = pd.read_csv(r"datasets\democracy-index-eiu.csv")
    di = _clean_index_df(
        df=di,
        country_col="Entity",
        year_col="Year",
        index_col="Democracy score",
        index=DI_COL
    )

    return di


def get_human_development_index() -> pd.DataFrame:
    hdi = pd.read_json(r"datasets\human_development_index.json")

    hdi = _clean_index_df(
        df=hdi,
        country_col="country",
        year_col="year",
        index_col="value",
        index=HDI_COL
    )
    
    return hdi


def _clean_index_df(df: pd.DataFrame, country_col, year_col: str, index_col: str, index: str):
    df.reset_index(drop=True, inplace=True)
    df.rename(
        columns={
            country_col: COUNTRY_COL,
            year_col: YEAR_COL,
            index_col: index
        },
        inplace=True
    )
    df = df[[COUNTRY_COL, YEAR_COL, index]]

    for col in df.columns:
        df.loc[:, col] = df[col].astype(COL_TYPE[col])
        df.loc[:, col] = COL_CONVERTER[col](df[col])

    df = df.dropna(how='any')

    df = df.sort_values([COUNTRY_COL, YEAR_COL], ascending=(True, True))

    df[index] = df[index].round(1)

    return df


def get_joined_di_hdi_df():
    di = get_democracy_index()
    hdi = get_human_development_index()

    df = di.merge(hdi, on=[COUNTRY_COL, YEAR_COL], how="inner")

    df.dropna(how='any', inplace=True)

    # Check if there are any NaN or zero values in the dataset
    if df.isna().any().any():
        raise ValueError("DataFrame contains NaN values after merging and cleaning.")
    if (df[[DI_COL, HDI_COL]] == 0).any().any():
        raise ValueError("0 found in index cols")

    return df
