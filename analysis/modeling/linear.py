"""
This fil e contains the functions for the analysis of the data.
It makes all the prints and plots that are needed to analyze the data for analysis.qmd
"""

import pandas as pd
import statsmodels.api as sm
from typing import List, Dict, Callable

from analysis.assumptions.linear import check_linear_regression_assumptions
from analysis.transformation import (
    get_independent_and_dependent_transformed_cols
)


def do_linear_regression(
        df: pd.DataFrame, 
        independent_cols: List[str],
        dependent_col: str,
        col_to_transform: Dict[str, Callable[[pd.Series], pd.Series]] = None,
        fit_kwargs: dict = None,
) -> sm.OLS:
    """
    Performs linear regression using statsmodels and returns the fitted model.
    """
    if not fit_kwargs:
        fit_kwargs = {
            "cov_type": "HC3"
        }
    

    df_copy = df.copy(deep=True)
    df_copy = df_copy[[*independent_cols, dependent_col]]
    df_copy = df_copy.dropna(how='any')

    x, y = get_independent_and_dependent_transformed_cols(
        df=df_copy,
        independent_cols=independent_cols,
        dependent_col=dependent_col,
        col_to_transform=col_to_transform
    )
    x = sm.add_constant(x)  # Adds intercept term
    model = sm.OLS(y, x).fit(**fit_kwargs)

    check_linear_regression_assumptions(
        model=model,
        independent_cols=x,
        dependent_col=y
    )

    print(model.summary())

    return model
