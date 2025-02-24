"""
This file contains the functions for the ARIMA analysis of the data.
It fits ARIMA models and performs diagnostics for analysis.qmd.
"""

import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.arima.model import ARIMA
from typing import Optional, Dict

def do_arima(
        df: pd.DataFrame,
        target_col: str,
        order: tuple = (1, 1, 1),
        exog_cols: Optional[Dict[str, pd.Series]] = None
) -> ARIMA:
    """
    Fits an ARIMA model using statsmodels and returns the fitted model.
    
    :param df: DataFrame containing the time series data.
    :param target_col: Column name for the dependent variable.
    :param order: Tuple specifying the ARIMA order (p, d, q).
    :param exog_cols: Optional dictionary of exogenous variables.
    :return: Fitted ARIMA model.
    """
    df_copy = df.copy(deep=True)
    df_copy = df_copy[[target_col] + (list(exog_cols.keys()) if exog_cols else [])]
    df_copy = df_copy.dropna(how='any')
    
    y = df_copy[target_col]
    exog = df_copy[exog_cols.keys()] if exog_cols else None
    
    model = ARIMA(y, order=order, exog=exog).fit()
    
    print(model.summary())
    
    return model
