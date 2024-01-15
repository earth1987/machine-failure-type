"""Custom code that gets used across more than notebook"""
import pandas as pd
from random import sample
import scipy
import numpy as np
from scipy.stats import f_oneway

def file_directory(TYPE):
  """Returns file directory given file status
  """
  if TYPE == 'raw':
      return 'C:/Users/billy/OneDrive/Documents/Python Scripts/1. Portfolio/machine-failure/data/raw/'
  elif TYPE == 'cleaned':
      return 'C:/Users/billy/OneDrive/Documents/Python Scripts/1. Portfolio/machine-failure/data/cleaned/'
  elif TYPE == 'processed':
      return 'C:/Users/billy/OneDrive/Documents/Python Scripts/1. Portfolio/machine-failure/data/processed/'

def perc_func(df, column):
    """Outputs the proportional breakdown of values for a given column in a dataframe."""
    summary = pd.DataFrame(
        dict(count=df[column].value_counts(),
             perc=round(df[column].value_counts(normalize=True)*100,1)
            )
    )
    # summary.loc['All'] = [df[column].value_counts().sum(), 
    #                       df[column].value_counts(normalize=True).sum()*100]

    return summary


def cramers_v(vars, df):
    """Compute cramers v between categorical variables for a given dataframe df.
    This calculation assumes each dataframe element consists of a (var1, var2) tuple.
    """
    var1, var2 = vars
    contigency_table = pd.crosstab(index=df[var1],
                                   columns=df[var2])
    X2 = scipy.stats.chi2_contingency(contigency_table)
    chi_stat = X2[0]
    N = len(df)
    min_dim = (min(contigency_table.shape)-1)
    return np.sqrt((chi_stat/N) / min_dim)


def oneway_anova(vars, df):
    """Compute one way ANOVA between a categorical and numeric variable for a given dataframe df.
    This calculation assumes each dataframe element consists of a (categorical var, numerical var) tuple.
    """
    cat, num = vars
    res = f_oneway(*tuple(
        [df[num].groupby(df[cat]).get_group(val) for val in df[cat].unique()]))
    return res.pvalue


def point_bs(vars, df):
    """Compute point biserial correlation between a dichotomous categorical variable and a numeric variable for a given dataframe df.
    This calculation assumes each dataframe element consists of a (categorical var, numerical var) tuple.
    """
    var1, var2 = vars
    res = scipy.stats.pointbiserialr(df[var1], df[var2])
    return res.statistic