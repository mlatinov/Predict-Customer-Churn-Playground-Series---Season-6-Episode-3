from functions.data_viz_f.viz_helpers import (
    num_summary,
    viz_histogram,
    viz_boxplot,
    viz_violin,
    viz_ECDF,
    categorical_summary,
    viz_bar
)
import pandas as pd

def univariate_eda(data: pd.DataFrame, x: str, y: str = None, color: str = None, stat: str = "percent", transformation: str = None) -> dict:
    """
    Perform univariate exploratory data analysis (EDA) for a given column.

    Generates summaries and plots based on the data type of the column.

    Parameters
    ----------
    data : pd.DataFrame
        The input DataFrame.
    x : str
        The column to analyze.
    y : str, optional
        Secondary column for bivariate plots (if applicable).
    color : str, optional
        Column for hue coloring in plots.
    stat : str, optional
        Statistic for bar plots ('count' or 'percent').
    transformation : str, optional
        Transformation to apply to numerical data.

    Returns
    -------
    dict
        A dictionary with 'summary' (list of DataFrames) and 'plots' (list of tuples).
    """
    plots = []
    summary = []
    if pd.api.types.is_object_dtype(data[x]):
        print("Categorical Entry")
        summary_cat = categorical_summary(
            data=data,
            columns=[x]
        )
        summary.append(summary_cat)

        plots.append(
            viz_bar(
                data=data,
                x=x,
                color=color,
                stat=stat
            )
        )

    elif pd.api.types.is_numeric_dtype(data[x]):
        print("Numerical Entry")
        summary_num = num_summary(
            data=data,
            columns=[x]
        )
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

    results = {
        "summary": summary,
        "plots": plots
    }
    return results