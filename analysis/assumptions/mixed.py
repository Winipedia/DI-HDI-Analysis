import statsmodels.api as sm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from pprint import pprint

from utils.util import suppress_output
from analysis.assumptions.linear import check_linear_regression_assumptions
from analysis.assumptions.base_check import check_assumptions, show_plot


def check_mixed_linear_regression_assumptions(
        model: sm.regression.mixed_linear_model.MixedLMResults,
        independent_cols: pd.DataFrame,
        dependent_col: pd.Series,
        group_col: pd.Series
):
    """
    Checks mixed model assumptions and prints diagnostic results.
    """

    linear_holds, linear_assumptions = suppress_output(check_linear_regression_assumptions)(
        model=model,
        independent_cols=independent_cols,
        dependent_col=dependent_col,
    )

    mixed_holds, mixed_assumptions = suppress_output(_check_additional_mixed_model_assumpitions)(
        model=model,
        independent_cols=independent_cols,
        dependent_col=dependent_col,
        group_col=group_col
    )

    assumptions_hold = linear_holds and mixed_holds
    assumptions = {
        **linear_assumptions,
        **mixed_assumptions
    }


    print(f"All assumptions hold: {assumptions_hold}")
    pprint(assumptions)

    return assumptions_hold, assumptions


def _check_additional_mixed_model_assumpitions(
        model: sm.regression.mixed_linear_model.MixedLMResults,
        independent_cols: pd.DataFrame,
        dependent_col: pd.Series,
        group_col: pd.Series
):
    return check_assumptions(
        assumption_funcs=[
            _check_normality_of_random_effects,
            _check_independence_of_random_effects_and_residuals,
            _check_homoscedasticity_of_random_effects,
        ],
        model=model,
        independent_cols=independent_cols,
        dependent_col=dependent_col,
        group_col=group_col
    )
    


def _check_normality_of_random_effects(model, **_):
    """
    Checks if the random effects are normally distributed using Shapiro-Wilk test and visualizations.
    """
    random_effects = model.random_effects
    re_values = np.array([re for group_re in random_effects.values() for re in group_re])
    sample_size = len(re_values)
    w_value, p_value = stats.shapiro(re_values)
    holds = (p_value > 0.05) or (w_value > 0.8 and sample_size > 100)
    
    message = (
        f"Shapiro-Wilk Test: W={w_value:.4f}, p={p_value:.4f}, Sample Size={sample_size}, "
        f"Normality Holds: {holds}"  
    )

    show_plot(
        plot_func=sns.histplot,
        plot_func_kwargs=dict(data=re_values, kde=True),
        message=message,
        title="Random Effects Distribution",
    )

    show_plot(
        plot_func=sm.qqplot,
        plot_func_kwargs=dict(data=re_values, line='s'),
        message=message,
        title="Q-Q Plot of Random Effects",
        pass_axis=True
    )
    
    return holds


def _check_independence_of_random_effects_and_residuals(
        model,
        **_
):
    """
    Checks if random effects are independent of residuals by computing correlation and visualizing it.
    """
    random_effects = model.random_effects
    residuals = model.resid
    
    re_values = np.array([re for group_re in random_effects.values() for re in group_re])
    correlation = np.corrcoef(re_values, residuals[:len(re_values)])[0, 1]
    holds = abs(correlation) < 0.1
    
    message = f"Correlation between Random Effects and Residuals: {correlation:.4f}, Independence Holds: {holds}"

    show_plot(
        plot_func=sns.scatterplot,
        plot_func_kwargs=dict(x=re_values, y=residuals[:len(re_values)]),
        message=message,
        title="Random Effects vs Residuals",
        y0line=True
    )
    
    return holds

def _check_homoscedasticity_of_random_effects(
        model,
        **_
):
    """
    Checks homoscedasticity of random effects by plotting group-level variance.
    """
    random_effects = model.random_effects
    
    group_variances = [np.var(list(re.values)) for re in random_effects.values()]
    variation_ratio = max(group_variances) / (min(group_variances) + 1e-6)
    holds = variation_ratio < 4  
    
    message = f"Max/Min Variance Ratio: {variation_ratio:.2f}, Homoscedasticity Holds: {holds}"  

    show_plot(
        plot_func=sns.boxplot,
        plot_func_kwargs=dict(y=group_variances),
        message=message,
        title="Variance of Random Effects Across Groups",
    )
    
    return holds
