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

    Parameters
    ----------
    data : pd.DataFrame
        The input DataFrame containing the numerical data.
    columns : List[str]
        List of numerical column names to summarize.

    Returns
    -------
    pd.DataFrame
        A DataFrame with rows for each input column and columns for each statistic:
        - count: Number of non-null values
        - mean: Arithmetic mean
        - median: 50th percentile
        - std: Standard deviation
        - min: Minimum value
        - max: Maximum value
        - quantile_25: 25th percentile (Q1)
        - quantile_75: 75th percentile (Q3)
        - skew: Skewness (distribution asymmetry)
        - kurtosis: Kurtosis (tail heaviness)
        - mean_null: Proportion of missing values (0-1 scale)

    Examples
    --------
    >>> df = pd.DataFrame({'age': [25, 30, 35, 40, 45], 'salary': [50000, 60000, 75000, 80000, 90000]})
    >>> summary = num_summary(df, ['age', 'salary'])

    Notes
    -----
    - All statistics are computed using pandas aggregation functions
    - Handles missing values appropriately (excluded from count, included in null proportion)
    - Useful for initial data exploration and identifying outliers via quantiles
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

    Parameters
    ----------
    data : pd.DataFrame
        The input DataFrame containing the data.
    x : str
        Column name for the variable to plot on x-axis.
    transformation : str, optional
        Transformation type to apply:
        - 'log': Log transformation (log base 10)
        - 'Box-Cox': Box-Cox power transformation
        - 'Yeo-Johnson': Yeo-Johnson transformation
        Default is None (no transformation).
    color : str, optional
        Column name for hue encoding to overlay distributions by category.
        Default is None.

    Returns
    -------
    Tuple[plt.Figure, plt.Axes]
        Tuple containing:
        - fig : matplotlib.figure.Figure
            The figure object
        - ax : matplotlib.axes.Axes
            The axes object for further customization

    Examples
    --------
    >>> fig, ax = viz_histogram(df, x='age', color='gender')
    >>> fig, ax = viz_histogram(df, x='salary', transformation='log')

    Notes
    -----
    - Uses seaborn with "Set2" palette and "whitegrid" style
    - Figure size is (6, 4) inches
    - Original data is copied to avoid modifying input DataFrame
    - Transformation is applied before plotting
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

    Parameters
    ----------
    data : pd.DataFrame
        The input DataFrame containing the data.
    x : str
        Column name for the primary axis variable.
    y : str, optional
        Column name for the secondary axis variable. Default is None.
    transformation : str, optional
        Transformation type ('log', 'Box-Cox', 'Yeo-Johnson'). Default is None.
    color : str, optional
        Column name for hue encoding to group data. Default is None.

    Returns
    -------
    Tuple[plt.Figure, plt.Axes]
        Tuple containing figure and axes objects.

    Examples
    --------
    >>> fig, ax = viz_boxplot(df, x='age', y='income', color='gender')

    Notes
    -----
    - Boxplot shows median (center line), quartiles (box), and outliers (points)
    - Q1 (25th percentile) marks the bottom of the box
    - Q3 (75th percentile) marks the top of the box
    - Whiskers extend to 1.5 * IQR from quartiles
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

    Parameters
    ----------
    data : pd.DataFrame
        The input DataFrame containing the data.
    x : str
        Column name for the primary axis variable.
    y : str
        Column name for the secondary axis variable.
    transformation : str, optional
        Transformation type ('log', 'Box-Cox', 'Yeo-Johnson'). Default is None.
    color : str, optional
        Column name for hue encoding to group data. Default is None.

    Returns
    -------
    Tuple[plt.Figure, plt.Axes]
        Tuple containing figure and axes objects.

    Examples
    --------
    >>> fig, ax = viz_violin(df, x='income', y='category', color='gender')

    Notes
    -----
    - Wider sections indicate higher probability density
    - Violin plot combines advantages of boxplot and kernel density estimation
    - Better for revealing multimodal distributions than boxplots
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

    Parameters
    ----------
    data : pd.DataFrame
        The input DataFrame containing the data.
    x : str
        Column name for the variable to plot.
    transformation : str, optional
        Transformation type ('log', 'Box-Cox', 'Yeo-Johnson'). Default is None.
    color : str, optional
        Column name for hue encoding to overlay multiple distributions. Default is None.

    Returns
    -------
    Tuple[plt.Figure, plt.Axes]
        Tuple containing figure and axes objects.

    Examples
    --------
    >>> fig, ax = viz_ECDF(df, x='salary', color='department')

    Notes
    -----
    - ECDF values range from 0 to 1 (0% to 100%)
    - Steeper slopes indicate more observations concentrated at those values
    - Non-parametric alternative to CDF assuming a specific distribution
    - Useful for comparing distributions without assumptions
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

    Parameters
    ----------
    data : pd.DataFrame
        The input DataFrame containing the categorical data.
    columns : List[str]
        List of categorical column names to summarize.

    Returns
    -------
    pd.DataFrame
        A DataFrame with rows for each input column and columns:
        - name: Column name
        - unique: Number of unique values
        - top_category: Most frequently occurring category
        - top_freq_%: Percentage frequency of top category (0-100)
        - missing: Percentage of missing values (0-100)

    Examples
    --------
    >>> df = pd.DataFrame({'color': ['red', 'blue', 'red', None, 'green']})
    >>> summary = categorical_summary(df, ['color'])

    Notes
    -----
    - All percentages are rounded to 3 decimal places
    - Missing values are excluded from unique count and frequency calculations
    - Useful for initial data quality assessment
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

    Parameters
    ----------
    data : pd.DataFrame
        The input DataFrame containing the data.
    x : str
        Column name for the categorical variable on x-axis.
    color : str, optional
        Column name for hue encoding to group bars by category. Default is None.
    stat : str, optional
        Statistic to display on y-axis ('count' or 'percent'). Default is 'percent'.

    Returns
    -------
    Tuple[plt.Figure, plt.Axes]
        Tuple containing figure and axes objects.

    Examples
    --------
    >>> fig, ax = viz_bar(df, x='category', stat='count')
    >>> fig, ax = viz_bar(df, x='category', color='gender', stat='percent')

    Notes
    -----
    - 'percent': Displays proportion as percentage
    - 'count': Displays absolute frequency counts
    - Figure size is (6, 4) inches
    - Uses seaborn with "Set2" palette
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

    Parameters
    ----------
    data : pd.DataFrame
        The input DataFrame containing the data.
    x : str
        Column name for the x-axis (independent variable).
    y : str
        Column name for the y-axis (dependent variable).
    color : str, optional
        Column name for hue encoding to color points by category. Default is None.
    transformation : str, optional
        Transformation type to apply to x ('log', 'Box-Cox', 'Yeo-Johnson').
        Default is None.

    Returns
    -------
    Tuple[plt.Figure, plt.Axes]
        Tuple containing figure and axes objects.

    Examples
    --------
    >>> fig, ax = viz_scatter(df, x='age', y='income', color='education')
    >>> fig, ax = viz_scatter(df, x='price', y='quantity', transformation='log')

    Notes
    -----
    - Each point represents one observation
    - Useful for identifying linear or non-linear relationships
    - Outliers appear as isolated points
    - Color encoding helps identify patterns within subgroups
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

    Creates a matrix of scatter plots showing relationships between all pairs of
    numerical variables, with distributions on the diagonal.

    Parameters
    ----------
    data : pd.DataFrame
        The input DataFrame containing the data.
    x : str
        Column name for transformation (if applicable).
    numerical_columns : List[str]
        List of numerical column names to include in the pairplot.
    transformation : str, optional
        Transformation type to apply ('log', 'Box-Cox', 'Yeo-Johnson'). Default is None.
    color : str, optional
        Column name for hue encoding to color by category. Default is None.

    Returns
    -------
    Tuple[plt.Figure, plt.Axes]
        Tuple containing figure and axes objects.

    Examples
    --------
    >>> fig, ax = viz_pairplot(df, x='age', numerical_columns=['age', 'income', 'experience'], color='gender')

    Notes
    -----
    - Diagonal plots show marginal distributions (density or histogram)
    - Off-diagonal plots show bivariate scatter plots
    - Useful for identifying multivariate relationships and outliers
    - Can become complex with many variables (recommend <5 variables)
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

    Parameters
    ----------
    data : pd.DataFrame
        The input DataFrame containing the data.
    x : str
        Column name for the x-axis variable.
    y : str
        Column name for the y-axis variable.
    transformation : str, optional
        Transformation type to apply to x ('log', 'Box-Cox', 'Yeo-Johnson').
        Default is None.

    Returns
    -------
    Tuple[plt.Figure, plt.Axes]
        Tuple containing figure and axes objects.

    Examples
    --------
    >>> fig, ax = viz_joint(df, x='age', y='income', transformation='log')

    Notes
    -----
    - Center plot is a hexbin plot (2D histogram) showing density
    - Top plot shows marginal distribution of x variable
    - Right plot shows marginal distribution of y variable
    - Hexbin bins improve visualization of overlapping points
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

    Parameters
    ----------
    data : pd.DataFrame
        The input DataFrame containing the data.
    x : str
        Column name for the first categorical variable (rows).
    y : str
        Column name for the second categorical variable (columns).

    Returns
    -------
    Tuple[plt.Figure, plt.Axes]
        Tuple containing figure and axes objects.

    Examples
    --------
    >>> fig, ax = viz_cross_tab(df, x='gender', y='purchased')

    Notes
    -----
    - Heatmap cells show count of observations
    - Darker colors indicate higher frequencies
    - Useful for identifying association between categorical variables
    - Can be combined with chi-square test for statistical inference
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

    Parameters
    ----------
    data : pd.DataFrame
        The input DataFrame containing the data.
    x : str
        Column name for the grouping/independent variable.
    y : str
        Column name for the target/dependent variable.
    test_kind : str
        Type of statistical test to perform:
        - 'Means': Compare means between groups
          * 2 groups: Independent t-test and Mann-Whitney U test
          * >2 groups: One-way ANOVA and Kruskal-Wallis test
        - 'Cramers': Association between categorical variables
          * Chi-square test and Cramer's V effect size

    Returns
    -------
    Dict[str, float]
        Dictionary containing test results:
        - For 'Means' (2 groups): {'One sample T-Test': p_value, 'One sample Man-Test': p_value}
        - For 'Means' (>2 groups): {'One Way Anova': p_value, 'Kruskal': p_value}
        - For 'Cramers': {'Chi Test': p_value, 'Cramers_V': effect_size}

    Raises
    ------
    ValueError
        If test_kind is not 'Means' or 'Cramers'.

    Examples
    --------
    >>> stats = stat_test_summary(df, x='group', y='score', test_kind='Means')
    >>> stats = stat_test_summary(df, x='gender', y='purchased', test_kind='Cramers')

    Notes
    -----
    Interpretation Guide:
    - P-values < 0.05 indicate significant difference/association (α=0.05)
    - Cramer's V ranges 0-1 (0=no association, 1=perfect association)
    - Parametric tests assume normality; non-parametric tests are robust to violations
    - Mann-Whitney U is non-parametric alternative to t-test (2 groups)
    - Kruskal-Wallis is non-parametric alternative to ANOVA (>2 groups)
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

