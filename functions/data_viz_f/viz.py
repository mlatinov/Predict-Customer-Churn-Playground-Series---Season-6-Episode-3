from functions.data_viz_f.viz_helpers import (
    num_summary,
    viz_histogram,
    viz_boxplot,
    viz_violin,
    viz_ECDF,
    categorical_summary,
    viz_bar,
    viz_scatter,
    viz_joint,
    viz_cross_tab,
    stat_test_summary
)
import pandas as pd
from typing import Dict, List, Optional, Tuple

def univariate_eda(
    data: pd.DataFrame,
    x: str,
    y: Optional[str] = None,
    color: Optional[str] = None,
    stat: str = "percent",
    transformation: Optional[str] = None
) -> Dict[str, List]:
    """
    Perform univariate exploratory data analysis (EDA) for a given column.

    Generates summaries and visualizations based on the data type of the column.
    For numerical columns, provides statistical summary and multiple distribution plots.
    For categorical columns, provides frequency summary and bar plot.

    Parameters
    ----------
    data : pd.DataFrame
        The input DataFrame containing the data to analyze.
    x : str
        The column name to analyze.
    y : str, optional
        Secondary column name for bivariate visualizations (e.g., boxplot grouping).
        Default is None.
    color : str, optional
        Column name for hue/color encoding in plots. Default is None.
    stat : str, optional
        Statistic to display in bar plots ('count' or 'percent'). Default is 'percent'.
    transformation : str, optional
        Transformation to apply to numerical data ('log', 'Box-Cox', 'Yeo-Johnson').
        Default is None.

    Returns
    -------
    Dict[str, List]
        Dictionary containing:
        - 'summary': List of DataFrames with statistical summaries
        - 'plots': List of tuples (fig, ax) containing matplotlib figure and axis objects

    Raises
    ------
    KeyError
        If column names do not exist in the DataFrame.
    TypeError
        If column type is neither numeric nor object (categorical).

    Examples
    --------
    >>> import pandas as pd
    >>> df = pd.read_csv('data.csv')
    >>> result = univariate_eda(df, x='age')
    >>> summary_df = result['summary'][0]
    >>> fig, ax = result['plots'][0]

    Notes
    -----
    - For numerical columns, generates: histogram, boxplot, violin plot, and ECDF
    - For categorical columns, generates: bar plot with frequency counts
    - All plots use seaborn styling with "Set2" palette
    """
    plots = []
    summary = []

    if pd.api.types.is_object_dtype(data[x]):
        print("Categorical Entry")
        summary_cat = categorical_summary(data=data, columns=[x])
        summary.append(summary_cat)

        plots.append(
            viz_bar(data=data, x=x, color=color, stat=stat)
        )

    elif pd.api.types.is_numeric_dtype(data[x]):
        print("Numerical Entry")
        summary_num = num_summary(data=data, columns=[x])
        summary.append(summary_num)

        plots.append(
            viz_histogram(
                data=data,
                x=x,
                transformation=transformation,
                color=color
            )
        )
        plots.append(
            viz_boxplot(
                data=data,
                x=x,
                y=y,
                transformation=transformation,
                color=color
            )
        )
        plots.append(
            viz_violin(
                data=data,
                x=x,
                y=y,
                transformation=transformation,
                color=color
            )
        )
        plots.append(
            viz_ECDF(
                data=data,
                x=x,
                transformation=transformation,
                color=color
            )
        )
    else:
        print("Error in type detection")

    results = {"summary": summary, "plots": plots}
    return results


def bivariate_eda(
    data: pd.DataFrame,
    x: str,
    y: str,
    color: Optional[str] = None,
    transformation: Optional[str] = None,
    test_kind: Optional[str] = None
) -> Dict[str, List]:
    """
    Perform bivariate exploratory data analysis (EDA) between two variables.

    Generates visualizations and statistical test summaries for relationships between
    two variables. Plot type depends on whether variables are numeric or categorical.

    Parameters
    ----------
    data : pd.DataFrame
        The input DataFrame containing the data to analyze.
    x : str
        The column name for the independent variable (x-axis/grouping variable).
    y : str
        The column name for the dependent variable (y-axis/target variable).
    color : str, optional
        Column name for hue/color encoding in plots. Default is None.
    transformation : str, optional
        Transformation to apply to numerical data ('log', 'Box-Cox', 'Yeo-Johnson').
        Default is None.
    test_kind : str, optional
        Type of statistical test to perform ('Means' or 'Cramers').
        - 'Means': Compares means using t-test (2 groups) or ANOVA (>2 groups)
        - 'Cramers': Tests association using Chi-square and Cramer's V
        Default is None (no statistical test performed).

    Returns
    -------
    Dict[str, List]
        Dictionary containing:
        - 'Plots': List of tuples (fig, ax) containing matplotlib figure and axis objects
        - 'Stat Test Summary': Dictionary with p-values and effect sizes

    Raises
    ------
    KeyError
        If column names do not exist in the DataFrame.

    Examples
    --------
    >>> import pandas as pd
    >>> df = pd.read_csv('data.csv')
    >>> result = bivariate_eda(df, x='age', y='income', test_kind='Means')
    >>> plots = result['Plots']
    >>> stats = result['Stat Test Summary']

    Notes
    -----
    - For numeric x: generates scatter plot and joint plot (hex)
    - For categorical x: generates cross-tabulation heatmap
    - Statistical tests use scipy.stats functions
    - Results include both parametric and non-parametric tests when applicable
    """
    plots = []
    summary = []

    # Check the type of x variable and generate appropriate visualizations
    if pd.api.types.is_numeric_dtype(data[x]):
        plots.append(
            viz_scatter(
                data=data,
                x=x,
                y=y,
                color=color,
                transformation=transformation
            )
        )
        plots.append(
            viz_joint(data=data, x=x, y=y, transformation=transformation)
        )
    elif pd.api.types.is_object_dtype(data[x]):
        plots.append(
            viz_cross_tab(data=data, x=x, y=y)
        )

    # Perform statistical test if specified
    if test_kind is not None:
        summary = stat_test_summary(
            data=data,
            x=x,
            y=y,
            test_kind=test_kind
        )

    results = {
        "Plots": plots,
        "Stat Test Summary": summary
    }
    return results
