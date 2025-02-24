import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
from typing import List, Dict, Callable

from analysis.assumptions.mixed import check_mixed_linear_regression_assumptions
from analysis.transformation import get_independent_and_dependent_transformed_cols


def do_mixed_linear_regression(
        df: pd.DataFrame,
        independent_cols: List[str],
        dependent_col: str,
        group_col: str,
        random_slope_cols: List[str] = None,
        col_to_transform: Dict[str, Callable[[pd.Series], pd.Series]] = None
) -> sm.regression.mixed_linear_model.MixedLMResults:
    """
    Performs a linear mixed-effects model using statsmodels and returns the fitted model.
    
    - `df`: DataFrame containing the data.
    - `independent_cols`: List of independent variable names.
    - `dependent_col`: Dependent variable name.
    - `group_col`: Column name for grouping (random effects).
    - `col_to_transform`: Optional dictionary mapping column names to transformation functions.
    """
    df_copy = df.copy(deep=True)
    df_copy = df_copy[[*independent_cols, dependent_col, group_col]]
    df_copy = df_copy.dropna(how='any')
    
    # Apply transformations if needed
    x, y = get_independent_and_dependent_transformed_cols(
        df=df_copy,
        independent_cols=independent_cols,
        dependent_col=dependent_col,
        col_to_transform=col_to_transform
    )
    group_col = df_copy[group_col]

    # Construct formula for mixed model
    fixed_effects = " + ".join(independent_cols)
    formula = f"{dependent_col} ~ {fixed_effects}"

    # Construct random slopes formula if specified
    re_formula = None
    if random_slope_cols:
        random_slopes = " + ".join(random_slope_cols)
        re_formula = f"~ {random_slopes}"  # Allow varying slopes per group

    # Fit the linear mixed model
    model = smf.mixedlm(formula, df_copy, groups=group_col, re_formula=re_formula)
    model = model.fit()

    # Check model assumptions
    check_mixed_linear_regression_assumptions(
        model=model,
        independent_cols=x,
        dependent_col=y,
        group_col=group_col
    )
    compute_r2_mixed(model)
    print(f"{model.aic=}")

    print(model.summary())

    return model
    

def compute_r2_mixed(model):
    """
    Computes Marginal R² (R²_m) and Conditional R² (R²_c) for a fitted mixed model.

    Parameters:
    - model: A fitted `MixedLMResults` object from statsmodels.

    Returns:
    - A tuple with Marginal R² (variance explained by fixed effects)
      and Conditional R² (variance explained by fixed + random effects).
    """

    # Variance explained by fixed effects (from fitted values)
    var_fixed = np.var(model.fittedvalues)

    # Variance explained by random effects
    var_random = model.cov_re.iloc[0, 0] if not model.cov_re.empty else 0

    # Residual variance (error variance)
    var_residual = model.scale

    # Compute R² values
    r2_marginal = round(var_fixed / (var_fixed + var_random + var_residual), 4)
    r2_conditional = round((var_fixed + var_random) / (var_fixed + var_random + var_residual), 4)

    print(f"Marginal R²: {r2_marginal}")
    print(f"Conditional R²: {r2_conditional}")

    return r2_marginal, r2_conditional
