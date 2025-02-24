import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from statsmodels.stats.stattools import durbin_watson
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.stats.diagnostic import linear_reset
from statsmodels.regression.linear_model import RegressionResultsWrapper
from scipy import stats

from analysis.assumptions.base_check import check_assumptions, show_plot


def check_linear_regression_assumptions(
        model: sm.OLS,
        independent_cols: pd.DataFrame,
        dependent_col: pd.Series,
):
    """
    Checks linear regression assumptions and displays each plot separately in Quarto.
    """

    return check_assumptions(
        assumption_funcs=[
            _check_linearity,
            _check_independence,
            _check_multicollinearity,
            _check_homoscedasticity,
            _check_normality_of_residuals,
            _check_outliers
        ],
        model=model,
        independent_cols=independent_cols,
        dependent_col=dependent_col
    )


def _check_multicollinearity(
        independent_cols: pd.DataFrame,
        **_
) -> bool:
    """
    Checks for multicollinearity using the correlation matrix.
    If any pairwise correlation exceeds 0.8, returns False.
    """

    if len(independent_cols.columns) <= 1:
        return True

    # Formal test
    independent_cols = independent_cols.drop(columns=['const'], errors='ignore')
    corr_matrix = independent_cols.corr()
    
    # Extract upper triangle without the diagonal
    upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

    # Count the number of highly correlated pairs
    high_corr = (upper_tri.abs() > 0.8).sum().sum()
    holds = high_corr == 0
    message = (
        f"Multicollinearity Check: {holds}. "
        f"Number of highly correlated pairs (|r| > 0.8, excluding self-correlations): {high_corr}."
    )

    show_plot(
        plot_func=sns.heatmap,
        plot_func_kwargs=dict(data=corr_matrix, annot=True, cmap='coolwarm', fmt='.2f'),
        title="Correlation Matrix",
        message=message,
    )
    
    return holds


def _check_linearity(
        model,
        independent_cols: pd.DataFrame, 
        dependent_col: pd.Series,
        **_
) -> bool:
    """
    Checks linearity by plotting scatter plots and computing correlation coefficients.
    """
    # Formal test
    independent_cols = independent_cols.drop(columns=['const'], errors='ignore')
    holds = 'Evaluate linearity assumption in plots'
    msg = ""
    if isinstance(model, RegressionResultsWrapper):
        reset_test = linear_reset(model, power=2, use_f=True)
        p_value = round(reset_test.pvalue, 4)
        holds = p_value > 0.05
        msg =  f" -> {p_value=}"
    
    msg = f"Linearity Assumption {holds=}" + msg
    
    # For each independent variable produce a scatter plot with an annotation.
    for col in independent_cols.columns:
        show_plot(
            plot_func=sns.scatterplot,
            plot_func_kwargs=dict(x=independent_cols[col], y=dependent_col),
            message=msg,
            title=f"Scatter Plot: {col} vs {dependent_col.name}",
        )
        
    # Final summary display for overall test.
    summary_msg = f"Overall Linearity Check: {holds=}"
    print(summary_msg)
    
    return holds


def _check_outliers(
        independent_cols: pd.DataFrame,
        **_
) -> bool:
    """
    Checks for outliers using boxplots and Z-scores.
    Returns False if extreme outliers are detected.
    """
    holds = True
    # Visual test: For each column display a boxplot with column-specific outlier info.
    independent_cols = independent_cols.drop(columns=['const'], errors='ignore')
    for col in independent_cols.columns:
        # Column-specific outlier count using z-scores
        col_z = np.abs(stats.zscore(independent_cols[col]))
        count = (col_z > 3).sum()
        sub_holds = count == 0
        msg = (
            f"""
            For '{col}': Outlier count (|z| > 3): {count}. 
            No Outliers Assumption {sub_holds=}
            """
        )
        
        show_plot(
            plot_func=sns.boxplot,
            plot_func_kwargs=dict(x=independent_cols[col]),
            message=msg,
            title=f"Boxplot: {col}"
        )

        holds = holds and sub_holds
    
    # Overall summary display
    summary_msg = f"Overall Outlier Check: {holds=}"
    print(summary_msg)
    
    return holds


def _check_homoscedasticity(
        model,
        **_
) -> bool:
    """
    Checks for homoscedasticity by plotting residuals vs fitted values.
    """
    # Formal test: Breusch-Pagan
    fitted_vals = model.fittedvalues
    residuals = model.resid
    _, pval, _, _ = het_breuschpagan(residuals, model.model.exog)
    holds = pval > 0.05  # High p-value means no heteroscedasticity
    message = (
        f"""
        Breusch-Pagan p-value: {pval:.3f}. 
        Homoscedasticity assumption {holds=}.
        """
    )

    show_plot(
        plot_func=sns.scatterplot,
        plot_func_kwargs=dict(x=fitted_vals, y=residuals),
        message=message,
        title="Residuals vs Fitted",
        y0line=True
    )
    
    return holds


def _check_normality_of_residuals(
        model,
        independent_cols: pd.DataFrame,
        **_
) -> bool:
    """
    Checks if residuals are normally distributed using histogram and Q-Q plot.
    """
    # Formal test: Shapiro-Wilk
    residuals = model.resid
    w_value, p_value = stats.shapiro(residuals)
    w_value, p_value = round(w_value, 4), round(p_value, 4)
    sample_size = model.nobs
    needed_size_for_assuming_normality = len(independent_cols.columns) * 30
    holds = (p_value > 0.05) or sample_size > needed_size_for_assuming_normality
    message = (
        f"""
        Shapiro-Wilk test {p_value=}; {w_value=}; {sample_size=}
        Residuals are normally distributed: {holds}
        If sample size is large enough (30 data points per predictor variable) then 
        the model is robust against deviations from normality.
        Long live the Central Limit Theorum!!!
        """
    )

    show_plot(
        plot_func=sns.histplot,
        plot_func_kwargs=dict(data=residuals, kde=True),
        message=message,
        title="Residuals Distribution"
    )

    show_plot(
        plot_func=sm.qqplot,
        plot_func_kwargs=dict(data=residuals, line='s'),
        message=message,
        title="Q-Q Plot",
        pass_axis=True,
    )
    
    return holds


def _check_independence(
        model,
        **_
) -> bool:
    """
    Checks the independence of residuals using Durbin-Watson statistic.
    """
    dw_stat = round(durbin_watson(model.resid), 4)
    holds = 1.5 < dw_stat < 2.5  # Values close to 2 indicate no autocorrelation
    message = (
        f"""
        Durbin-Watson statistic: {dw_stat}. 
        Independence assumption {holds=}
        """
    )

    show_plot(
        plot_func=plot_acf,
        plot_func_kwargs=dict(x=model.resid),
        message=message,
        title="Residuals Over Time",
        pass_axis=True,
    )
    
    return holds
