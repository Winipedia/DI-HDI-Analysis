"""
Contains functions that transform data for models using pandas
"""

import pandas as pd
from scipy.stats import boxcox, yeojohnson
import numpy as np
from typing import Dict, Callable, Tuple, List


def get_independent_and_dependent_transformed_cols(
        df: pd.DataFrame,
        independent_cols: List[str],
        dependent_col: str,
        col_to_transform: Dict[str, Callable[[pd.Series], pd.Series]] = None,
) -> Tuple[pd.DataFrame, pd.Series]:
    
    if col_to_transform is None:
        col_to_transform = {}

    for col, transform in col_to_transform.items():
        col_transform_name = f"{col}_{transform.__name__}"
        df[col_transform_name] = transform(df[col])
        if col in independent_cols:
            independent_cols.remove(col)
            independent_cols.append(col_transform_name)
        elif col == dependent_col:
            dependent_col = col_transform_name
        else:
            raise ValueError(F"{col} not a dependent or independent variable")
    
    independent_cols = df[independent_cols]
    dependent_col = df[dependent_col]
    return independent_cols, dependent_col


def log_transform(x: pd.Series, base: float = np.e, scale: float = 1) -> pd.Series:
    """Applies a light log transformation to x with a specified base, adding 0.1 to avoid log(0).
    
    A scale factor is included to control the strength of the transformation.
    """
    return np.log1p(scale * x) / np.log(base) / scale


def box_cox_transform(x: pd.Series) -> pd.Series:
    """Applies Box-Cox transformation to x"""
    return pd.Series(boxcox(x.dropna())[0], index=x.dropna().index)


def yeo_johnson_transform(x: pd.Series) -> pd.Series:
    """Applies Yeo-Johnson transformation to x"""
    return pd.Series(yeojohnson(x.dropna())[0], index=x.dropna().index)


def root_transform(x: pd.Series, root: float = 2) -> pd.Series:
    """Applies root transformation to x"""
    return x.pow(1 / root)


def inverse_transform(x: pd.Series) -> pd.Series:
    """Applies inverse transformation to x"""
    return x.apply(lambda v: 1 / v if v != 0 else np.nan)


def z_score_transform(x: pd.Series) -> pd.Series:
    """Applies z-score transformation to x"""
    return (x - x.mean()) / x.std()


def min_max_transform(x: pd.Series) -> pd.Series:
    """Applies min-max transformation to x"""
    return (x - x.min()) / (x.max() - x.min())


def power_transform(x: pd.Series, pow: int) -> pd.Series:
    """Applies exponential transformation to x"""
    return x.pow(pow)
