"""
Data preparation and feature engineering utilities for customer churn prediction.

This module contains functions for loading, transforming, and engineering features
from customer data for machine learning modeling purposes.
"""
from matplotlib.pyplot import axis
import pandas as pd
from scipy import stats
import numpy as np
from sklearn.model_selection import train_test_split

def load_data_sample(url: str, n: int) -> pd.DataFrame:
    """
    Load a random sample of rows from a CSV file for pipeline prototyping.

    Parameters
    ----------
    url : str
        Path or URL to the CSV file.
    n : int
        Number of rows to sample.

    Returns
    -------
    pd.DataFrame
        A DataFrame containing n randomly sampled rows (random_state=42 for reproducibility).
    """
    data_sample = pd.read_csv(url).sample(n=n, random_state=42)
    return data_sample

def data_split(train_url, train_size, test_size) : 
    """
    Function to take data paths and return a X_train Y_train X_test, Y_test
    """
    # Load the data 
    train_data = pd.read_csv(train_url)
    X = train_data.drop("Churn",axis = 1)
    y = train_data["Churn"]

    # Split the data with sklearn 
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        random_state=42,
        test_size=test_size,
        train_size=train_size
    )
    data_splits = {
        "x_train" : X_train,
        "y_train" : y_train,
        "x_test"  : X_test,
        "y_test"  : y_test
    }
    
    return data_splits

def mutate_payment(data: pd.DataFrame) -> pd.DataFrame:
    """
    Create automatic payment flag and normalize payment method labels.

    Extracts automatic payment information into a binary feature and removes
    the "(automatic)" suffix from payment method labels for cleaner categorization.

    Parameters
    ----------
    data : pd.DataFrame
        Customer data with PaymentMethod column. Modified in place.

    Returns
    -------
    pd.DataFrame
        The same DataFrame with new 'automatic_payment' column and cleaned PaymentMethod values.

    Notes
    -----
    This function modifies the input DataFrame in place.
    """
    # Create binary feature: 1 if payment is automatic, 0 otherwise
    data["automatic_payment"] = data["PaymentMethod"].isin([
        "Bank transfer (automatic)",
        "Credit card (automatic)"
    ]).astype(int)

    # Standardize payment method names by removing "(automatic)" suffix
    data["PaymentMethod"] = data["PaymentMethod"].replace({
        "Bank transfer (automatic)": "Bank transfer",
        "Credit card (automatic)": "Credit card"
    })
    return data


# ============================================================================
# TENURE-BASED FEATURES
# ============================================================================

def mutate_customer_lifetime_buckets(data: pd.DataFrame) -> pd.DataFrame:
    """
    Categorize customer tenure into meaningful lifecycle buckets.

    Segments customer tenure into 5 distinct buckets representing different
    lifecycle stages: new, early, established, loyal, and veteran customers.

    Parameters
    ----------
    data : pd.DataFrame
        Customer data with 'tenure' column (in months).

    Returns
    -------
    pd.DataFrame
        DataFrame with new 'customer_lifetime_buckets' ordinal category column.
    """
    # Bin tenure into meaningful customer lifecycle segments
    data["customer_lifetime_buckets"] = pd.cut(
        data["tenure"],
        bins=[-1, 5, 10, 20, 40, float("inf")],
        labels=["0-5", "5-10", "10-20", "20-40", "60+"]
    )
    return data


def mutate_new_customer_flag(data: pd.DataFrame) -> pd.DataFrame:
    """
    Create binary flag identifying newly acquired customers.

    Flags customers within their first 6 months as new, useful for analyzing
    early churn patterns and onboarding effectiveness.

    Parameters
    ----------
    data : pd.DataFrame
        Customer data with 'tenure' column (in months).

    Returns
    -------
    pd.DataFrame
        DataFrame with new 'new_customer_flag' binary column (1 = new customer).
    """
    # Flag customers with 6 months or less tenure as new
    data["new_customer_flag"] = (data["tenure"] <= 6).astype(int)
    return data

# ============================================================================
# VALUE AND PRICING SIGNALS
# ============================================================================

def mutate_value_gap(data: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate pricing discrepancy relative to customer tenure.

    Computes the difference between monthly charges and the average monthly cost
    implied by total tenure, capturing whether customers are paying more or less
    than expected based on their tenure length.

    Parameters
    ----------
    data : pd.DataFrame
        Customer data with 'MonthlyCharges' and 'tenure' columns.

    Returns
    -------
    pd.DataFrame
        DataFrame with new 'value_gap' column (pricing signal).

    Notes
    -----
    Positive values indicate current charges exceed historical average;
    negative values indicate customers are paying less than their tenure average.
    """
    # Calculate expected average monthly charge based on tenure
    expected_total = data["MonthlyCharges"] * data["tenure"]
    # Compare current monthly charge against historical average
    data["value_gap"] = data["MonthlyCharges"] - expected_total
    return data


# ============================================================================
# SERVICE RICHNESS SIGNALS
# ============================================================================

def mutate_count_services(data: pd.DataFrame) -> pd.DataFrame:
    """
    Count the total number of services subscribed by each customer.

    Aggregates all service subscriptions (phone, internet, security, backup, etc.)
    into a single count metric. Useful for understanding customer engagement
    and service portfolio depth.

    Parameters
    ----------
    data : pd.DataFrame
        Customer data with service columns (PhoneService, InternetService,
        OnlineSecurity, OnlineBackup, DeviceProtection, TechSupport,
        StreamingTV, StreamingMovies, MultipleLines).

    Returns
    -------
    pd.DataFrame
        DataFrame with new 'count_services' column containing the total
        number of services (0-9) for each customer.

    Notes
    -----
    Counts any column marked as "Yes" in the services list. Assumes binary
    Yes/No values in service columns.
    """
    # List of all available services to aggregate
    total_services = [
        "PhoneService", "MultipleLines", "InternetService", "OnlineSecurity",
        "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV",
        "StreamingMovies"
    ]
    # Sum the number of "Yes" values across all service columns
    data["count_services"] = (data[total_services] == "Yes").sum(axis=1)
    return data


def mutate_premium_user_flag(data: pd.DataFrame) -> pd.DataFrame:
    """
    Identify high-value customers based on service portfolio depth.

    Creates a binary flag for customers subscribed to 7 or more services,
    indicating premium/high-engagement users who likely contribute more
    to revenue and may have different churn patterns.

    Parameters
    ----------
    data : pd.DataFrame
        Customer data with 'count_services' column (typically created by
        mutate_count_services).

    Returns
    -------
    pd.DataFrame
        DataFrame with new 'premium_user_flag' binary column (1 = premium user).

    Notes
    -----
    Requires 'count_services' column to exist. Threshold of 7 services
    represents subscription to 78%+ of available services.
    """
    # Flag customers with 7+ services as premium users
    data["premium_user_flag"] = (data["count_services"] >= 7).astype(int)
    return data

def mutate_model_clean_data(data) : 
    """
    Function to clean and transform the the data for modeling 
    """
    # Columns that are Yes and No encoded 
    columns = ["Partner","Dependents","PhoneService","PaperlessBilling"]
    data[columns] = data[columns].apply(lambda col : col.map({"Yes" : 1, "No" : 0}))
    # Mutate Gender as 0 : Female and 1 : Male
    data["gender"] = data["gender"].replace({"Female" : 0, "Male" : 1})
    # Mutate Churn to be 0 and 1 
    data["Churn"] = data["Churn"].replace({"Yes" : 1, "No" : 0})
    return data

def column_transform(data: pd.DataFrame, column: str, transformation: str) -> pd.DataFrame:
    """
    Transform a column in the DataFrame, used primarily for plotting choices.

    Parameters
    ----------
    data : pd.DataFrame
        The input DataFrame.
    column : str
        The column to transform.
    transformation : str
        The type of transformation ('log', 'Box-Cox', 'Yeo').

    Returns
    -------
    pd.DataFrame
        The DataFrame with the transformed column.
    """
    data = data.copy()
    if transformation == "log":
        data[column] = np.log(data[column])
    elif transformation == "Box-Cox":
        data[column], _ = stats.boxcox(data[column])
    elif transformation == "Yeo":
        data[column], _ = stats.yeojohnson(data[column])
    else:
        print(f"No transformation applied: {transformation}")
    return data
