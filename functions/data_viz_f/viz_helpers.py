import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
import seaborn as sns
from functions.data_prep_f.data_helpers import column_transform

#### =========== Function to univaraite Analysis Numerical ==============================
data =  pd.read_csv("sample_data/train.csv")

def num_summary(data: pd.DataFrame, columns: list) -> pd.DataFrame:
    """
    Provide and return tabular summarization of numerical columns.

    Parameters
    ----------
    data : pd.DataFrame
        The input DataFrame containing the data.
    columns : list
        List of column names to summarize.

    Returns
    -------
    pd.DataFrame
        A DataFrame with summary statistics for each column.
    """
    stats_df = data[columns].agg([
        "count", "mean", "median", "std", "min", "max",
        lambda x: x.quantile(0.25),
        lambda x: x.quantile(0.75),
        lambda x: x.skew(),
        lambda x: x.kurtosis(),
        lambda x: x.isnull().mean()
    ]).T
    stats_df.columns = [
        "count", "mean", "median", "std", "min", "max", "quantile_25",
        "quantile_75", "skew", "kurtosis", "mean_null"
    ]
    return stats_df

def viz_histogram(data: pd.DataFrame, x: str, transformation: str = None, color: str = None) -> tuple:
    """
    Plot a histogram with optional transformations.

    Parameters
    ----------
    data : pd.DataFrame
        The input DataFrame.
    x : str
        Column name for the x-axis.
    transformation : str, optional
        Transformation to apply ('log', 'Box-Cox', 'Yeo').
    color : str, optional
        Column name for hue coloring.

    Returns
    -------
    tuple
        Figure and axis objects.
    """
    # Apply the transformation if needed
    if transformation is not None:
        data = data.copy()
        data = column_transform(
            data=data,
            column=x,
            transformation=transformation
        )
    # Plot the histogram
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.histplot(
        data=data,
        x=x,
        ax=ax,
        hue=color
    )
    plt.title(f"Histogram of {x}")
    plt.show()
    return fig, ax

def viz_boxplot(data: pd.DataFrame, x: str, y: str = None, transformation: str = None, color: str = None) -> tuple:
    """
    Plot a boxplot with optional transformations.

    Parameters
    ----------
    data : pd.DataFrame
        The input DataFrame.
    x : str
        Column name for the x-axis.
    y : str, optional
        Column name for the y-axis.
    transformation : str, optional
        Transformation to apply.
    color : str, optional
        Column name for hue coloring.

    Returns
    -------
    tuple
        Figure and axis objects.
    """
    # Apply the transformation if needed
    if transformation is not None:
        data = data.copy()
        data = column_transform(
            data=data,
            column=x,
            transformation=transformation
        )
    # Plot the boxplot
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.boxplot(
        data=data,
        x=x,
        y=y,
        ax=ax,
        hue=color
    )
    plt.title(f"Boxplot of {x}")
    plt.show()
    return fig, ax

def viz_violin(data: pd.DataFrame, x: str, y: str, transformation: str = None, color: str = None) -> tuple:
    """
    Plot a violin plot with optional transformations.

    Parameters
    ----------
    data : pd.DataFrame
        The input DataFrame.
    x : str
        Column name for the x-axis.
    y : str
        Column name for the y-axis.
    transformation : str, optional
        Transformation to apply.
    color : str, optional
        Column name for hue coloring.

    Returns
    -------
    tuple
        Figure and axis objects.
    """
    # Apply transformation if needed
    if transformation is not None:
        data = data.copy()
        data = column_transform(
            data=data,
            column=x,
            transformation=transformation
        )
    # Plot a violin plot
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.violinplot(
        data=data,
        x=x,
        y=y,
        ax=ax,
        hue=color
    )
    plt.show()
    return fig, ax

def viz_ECDF(data: pd.DataFrame, x: str, transformation: str = None, color: str = None) -> tuple:
    """
    Plot an Empirical Cumulative Distribution Function (ECDF) with optional transformations.

    Parameters
    ----------
    data : pd.DataFrame
        The input DataFrame.
    x : str
        Column name for the x-axis.
    transformation : str, optional
        Transformation to apply.
    color : str, optional
        Column name for hue coloring.

    Returns
    -------
    tuple
        Figure and axis objects.
    """
    # Apply transformation if needed
    if transformation is not None:
        data = data.copy()
        data = column_transform(
            data=data,
            column=x,
            transformation=transformation
        )
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.ecdfplot(
        data=data,
        x=x,
        ax=ax,
        hue=color
    )
    plt.show()
    return fig, ax

# =================== Univaraite Analysis Categorical Features ==================

def categorical_summary(data: pd.DataFrame, columns: list) -> pd.DataFrame:
    """
    Provide a summary for categorical columns.

    Parameters
    ----------
    data : pd.DataFrame
        The input DataFrame.
    columns : list
        List of categorical column names.

    Returns
    -------
    pd.DataFrame
        A DataFrame with summary statistics for each column.
    """
    summary = []
    for col in columns:
        col_summary = {
            "name": col,
            "unique": data[col].nunique(),
            "top_category": data[col].value_counts().idxmax(),
            "top_freq_%": float(round(data[col].value_counts(normalize=True).iloc[0] * 100, 3)),
            "missing": float(round(data[col].isnull().mean() * 100, 3))
        }
        summary.append(col_summary)
    stats_df = pd.DataFrame(summary)
    return stats_df

def viz_bar(data: pd.DataFrame, x: str, color: str = None, stat: str = "percent") -> tuple:
    """
    Plot a bar plot for categorical data.

    Parameters
    ----------
    data : pd.DataFrame
        The input DataFrame.
    x : str
        Column name for the x-axis.
    color : str, optional
        Column name for hue coloring.
    stat : str, optional
        Statistic to display ('count' or 'percent').

    Returns
    -------
    tuple
        Figure and axis objects.
    """
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.countplot(
        data=data,
        x=x,
        ax=ax,
        hue=color,
        stat=stat
    )
    plt.show()
    return fig, ax  
