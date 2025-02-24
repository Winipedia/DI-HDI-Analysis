import os
import sys
import pandas as pd
import country_converter as coco


def suppress_output(func):
    def wrapper(*args, **kwargs):
        with open(os.devnull, 'w') as fnull:
            old_stdout = sys.stdout
            old_stderr = sys.stderr
            sys.stdout = fnull  # Redirect stdout
            sys.stderr = fnull  # Redirect stderr
            try:
                return func(*args, **kwargs)
            finally:
                sys.stdout = old_stdout  # Restore stdout
                sys.stderr = old_stderr  # Restore stderr
    return wrapper


@suppress_output
def convert_country_col_to_iso3(co_col: pd.Series) -> pd.Series:
    co_col_iso3: list = coco.convert(co_col, to="ISO3", not_found="")
    co_col_iso3 = pd.Series(co_col_iso3, dtype=str)

    co_col_iso3 = co_col_iso3.replace("", pd.NA)

    return co_col_iso3
