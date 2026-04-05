from scipy.stats import chi2_contingency
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
import seaborn as sns
from typing import Tuple, Optional, Dict, List
from functions.data_prep_f.data_helpers import column_transform

# =========== Univariate Analysis: Numerical Features ==========================

def num_summary(data: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    """
    Provide and return tabular summarization of numerical columns.

    Computes comprehensive descriptive statistics for numerical columns including
    measures of central tendency, dispersion, shape, and data quality.
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


def viz_histogram(
    data: pd.DataFrame,
    x: str,
    transformation: Optional[str] = None,
    color: Optional[str] = None
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plot a histogram with optional transformations for numerical data.

    Displays the distribution of a numerical variable with optional color encoding
    by a categorical variable. Data transformations (log, Box-Cox, Yeo-Johnson)
    can be applied to improve visualization of skewed distributions.
    """
    # Apply the transformation if specified
    if transformation is not None:
        data = data.copy()
        data = column_transform(
            data=data,
            column=x,
            transformation=transformation
        )

    # Configure theme and create figure
    sns.set_theme(palette="Set2", style="whitegrid")
    fig, ax = plt.subplots(figsize=(6, 4))

    # Plot histogram
    sns.histplot(
        data=data,
        x=x,
        ax=ax,
        hue=color
    )
    ax.set_title(f"Histogram of {x}")
    plt.tight_layout()
    fig.show()

    return fig, ax


def viz_boxplot(
    data: pd.DataFrame,
    x: str,
    y: Optional[str] = None,
    transformation: Optional[str] = None,
    color: Optional[str] = None
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plot a boxplot with optional transformations for numerical data.

    Displays the distribution and outliers of a numerical variable, optionally
    grouped by categorical variables. Useful for comparing distributions across
    groups and identifying outliers.
    """
    # Apply the transformation if specified
    if transformation is not None:
        data = data.copy()
        data = column_transform(
            data=data,
            column=x,
            transformation=transformation
        )

    # Configure theme and create figure
    sns.set_theme(palette="Set2", style="whitegrid")
    fig, ax = plt.subplots(figsize=(6, 4))

    # Plot boxplot
    sns.boxplot(
        data=data,
        x=x,
        y=y,
        ax=ax,
        hue=color
    )
    ax.set_title(f"Boxplot of {x}")
    plt.tight_layout()
    fig.show()

    return fig, ax


def viz_violin(
    data: pd.DataFrame,
    x: str,
    y: str,
    transformation: Optional[str] = None,
    color: Optional[str] = None
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plot a violin plot with optional transformations for numerical data.

    Displays the full probability density function of a numerical variable
    as a mirrored histogram. Useful for comparing distribution shapes across
    multiple groups.
    """
    # Apply transformation if specified
    if transformation is not None:
        data = data.copy()
        data = column_transform(
            data=data,
            column=x,
            transformation=transformation
        )

    # Configure theme and create figure
    sns.set_theme(palette="Set2", style="whitegrid")
    fig, ax = plt.subplots(figsize=(6, 4))

    # Plot violin plot
    sns.violinplot(
        data=data,
        x=x,
        y=y,
        ax=ax,
        hue=color
    )
    ax.set_title(f"Violin Plot of {x} with transformation: {transformation}")
    plt.tight_layout()
    fig.show()

    return fig, ax


def viz_ECDF(
    data: pd.DataFrame,
    x: str,
    transformation: Optional[str] = None,
    color: Optional[str] = None
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plot an Empirical Cumulative Distribution Function (ECDF) with optional transformations.

    Displays the proportion of observations less than or equal to each value.
    Useful for comparing distribution shapes and identifying quantiles without
    assuming a specific distribution.
    """
    # Apply transformation if specified
    if transformation is not None:
        data = data.copy()
        data = column_transform(
            data=data,
            column=x,
            transformation=transformation
        )

    # Configure theme and create figure
    sns.set_theme(palette="Set2", style="whitegrid")
    fig, ax = plt.subplots(figsize=(6, 4))

    # Plot ECDF
    sns.ecdfplot(
        data=data,
        x=x,
        ax=ax,
        hue=color
    )
    ax.set_title(f"ECDF Plot of {x} with transformation: {transformation}")
    plt.tight_layout()
    fig.show()

    return fig, ax


# =================== Univariate Analysis: Categorical Features ==================

def categorical_summary(data: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    """
    Provide a summary for categorical columns.

    Computes key statistics for categorical variables including unique values,
    frequency of the most common category, and missing value proportion.
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


def viz_bar(
    data: pd.DataFrame,
    x: str,
    color: Optional[str] = None,
    stat: str = "percent"
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plot a bar plot for categorical data.

    Displays the frequency or count of each category, optionally grouped
    by an additional categorical variable.
    """
    # Configure theme and create figure
    sns.set_theme(palette="Set2", style="whitegrid")
    fig, ax = plt.subplots(figsize=(6, 4))

    # Plot bar chart
    sns.countplot(
        data=data,
        x=x,
        ax=ax,
        hue=color,
        stat=stat
    )
    ax.set_title(f"Bar Plot of {x}")
    plt.tight_layout()
    fig.show()

    return fig, ax


# =================== Bivariate Analysis Functions ===============================

def viz_scatter(
    data: pd.DataFrame,
    x: str,
    y: str,
    color: Optional[str] = None,
    transformation: Optional[str] = None
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plot a scatter plot for bivariate numerical data.

    Displays the relationship between two numerical variables, with optional
    color encoding for a third categorical variable.

    """
    # Apply transformations if needed
    if transformation is not None:
        data = data.copy()
        data = column_transform(
            data=data,
            column=x,
            transformation=transformation
        )

    # Configure theme and create figure
    sns.set_theme(palette="Spectral", style="whitegrid")
    fig, ax = plt.subplots(figsize=(6, 4))

    # Plot scatter plot
    sns.scatterplot(
        data=data,
        x=x,
        y=y,
        ax=ax,
        hue=color
    )
    ax.set_title(f"Scatter Plot of {x} vs {y} with transformation: {transformation}")
    plt.tight_layout()
    fig.show()

    return fig, ax


def viz_pairplot(
    data: pd.DataFrame,
    x: str,
    numerical_columns: List[str],
    transformation: Optional[str] = None,
    color: Optional[str] = None
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plot a pairplot to visualize pairwise relationships among numerical variables.

    """
    # Select relevant columns
    if color is not None:
        subset_data = data[numerical_columns + [color]]
    else:
        subset_data = data[numerical_columns]

    # Apply transformation if needed
    if transformation is not None:
        subset_data = column_transform(
            data=subset_data,
            column=x,
            transformation=transformation
        )

    # Configure theme and create pairplot
    sns.set_theme(palette="Spectral", style="whitegrid")
    fig = sns.pairplot(
        subset_data,
        hue=color,
        plot_kws=dict(marker="+", linewidth=1),
        diag_kws=dict(fill=False)
    )
    fig.fig.suptitle(f"Pairplot of {numerical_columns}", y=1.01)
    plt.tight_layout()
    fig.show()

    return fig, fig.axes[0]

def viz_joint(
    data: pd.DataFrame,
    x: str,
    y: str,
    transformation: Optional[str] = None
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plot a joint plot showing bivariate distribution with marginal distributions.

    Combines a bivariate plot with marginal univariate plots to show the full
    relationship between two variables along with their individual distributions.
    """
    # Apply transformation if needed
    if transformation is not None:
        data = data.copy()
        data = column_transform(
            data=data,
            column=x,
            transformation=transformation
        )

    # Configure theme and create joint plot
    sns.set_theme(palette="Spectral", style="whitegrid")
    fig = sns.jointplot(
        data=data,
        x=x,
        y=y,
        kind="hex"
    )
    fig.fig.suptitle(f"Joint Plot of {x} and {y}", y=1.01)
    plt.tight_layout()
    fig.show()

    return fig, fig.ax_joint


def viz_cross_tab(
    data: pd.DataFrame,
    x: str,
    y: str
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plot a cross-tabulation heatmap showing relationship between categorical variables.

    Displays a contingency table as a heatmap with color intensity representing
    frequency or proportion of observations in each category combination.
    """

    # Create cross-tabulation
    data_cross = pd.crosstab(data[x], data[y])

    # Configure theme and create heatmap
    sns.set_theme(palette="Spectral", style="whitegrid")
    fig, ax = plt.subplots(figsize=(6, 4))

    sns.heatmap(
        data=data_cross,
        ax=ax,
        annot=True,
        fmt='d',
        cmap='YlOrRd'
    )
    ax.set_title(f"Cross-tabulation Heatmap: {x} vs {y}")
    plt.tight_layout()
    fig.show()

    return fig, ax


def stat_test_summary(
    data: pd.DataFrame,
    x: str,
    y: str,
    test_kind: str
) -> Dict[str, float]:
    """
    Perform statistical tests to assess relationships between variables.

    Conducts appropriate statistical tests based on variable types and number of groups.
    Includes both parametric and non-parametric alternatives.

    """
    # Test for comparing means across groups
    if test_kind == "Means":
        # Group data by x variable
        groups = [group[y].values for _, group in data.groupby(x)]

        # Two-group comparison
        if data[x].nunique() == 2:
            # Parametric: Independent samples t-test
            param_test, p_param = stats.ttest_ind(*groups)
            # Non-parametric: Mann-Whitney U test
            non_param, p_non_param = stats.mannwhitneyu(*groups)
            summary = {
                "One sample T-Test": p_param,
                "One sample Man-Test": p_non_param
            }
        # Multiple group comparison
        elif data[x].nunique() > 2:
            # Parametric: One-way ANOVA
            param_test, p_param = stats.f_oneway(*groups)
            # Non-parametric: Kruskal-Wallis test
            non_param, p_non_param = stats.kruskal(*groups)
            summary = {
                "One Way Anova": p_param,
                "Kruskal": p_non_param
            }
        else:
            print("Error in test_kind == Means: Insufficient groups")
            summary = {}

    # Test for association between categorical variables
    elif test_kind == "Cramers":
        # Create contingency table
        groups = pd.crosstab(data[x], data[y])
        # Chi-square test
        chi2, p, dof, expected = chi2_contingency(groups)
        # Cramer's V: effect size (0=no association, 1=perfect association)
        n = groups.values.sum()
        cramers_v = np.sqrt(chi2 / (n * (min(groups.shape) - 1)))
        summary = {
            "Chi Test": p,
            "Cramers_V": cramers_v
        }
    else:
        print(f"Wrong Test Kind: {test_kind}. Use 'Means' or 'Cramers'")
        summary = {}

    return summary

