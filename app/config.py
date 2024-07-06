import warnings
import pandas as pd


def turn_off_warnings():
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    warnings.filterwarnings("ignore", category=UserWarning)
    pd.set_option('display.max_columns', None)
    # pd.set_option('display.max_rows', None)
    # pd.set_option('future.no_silent_downcasting', True)
    # pd.set_option("display.float_format", "{:.6f}".format)
