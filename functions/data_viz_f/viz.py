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
from typing import Dict, List, Optional

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
