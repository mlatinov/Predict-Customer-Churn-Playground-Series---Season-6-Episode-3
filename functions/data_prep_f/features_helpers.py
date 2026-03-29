"""
Feature engineering pipeline orchestration for customer churn prediction.

This module provides high-level pipelines that combine multiple feature
transformations from data_helpers into cohesive feature engineering workflows.
Three pipelines are available with varying levels of feature coverage.
"""

import pandas as pd

from functions.data_prep_f.data_helpers import (
    mutate_payment,
    mutate_customer_lifetime_buckets,
    mutate_new_customer_flag,
    mutate_value_gap,
    mutate_count_services,
    mutate_premium_user_flag,
)

# ============================================================================
# TENURE-FOCUSED FEATURE ENGINEERING PIPELINE
# ============================================================================

def tenure_feature_eng(data):
    """
    Create tenure-based features for customer lifecycle analysis.

    Applies a focused feature engineering pipeline that extracts payment patterns
    and segments customers by tenure. Ideal for analyzing customer lifecycle
    stages and churn patterns across different tenure groups.

    Parameters
    ----------
    data : pd.DataFrame
        Raw customer data with PaymentMethod and tenure columns.

    Returns
    -------
    pd.DataFrame
        Enriched DataFrame with features:
        - automatic_payment: Binary flag for automatic payments
        - customer_lifetime_buckets: Tenure segmentation (0-5, 5-10, 10-20, 20-40, 60+)
        - new_customer_flag: Binary flag for customers with <=6 months tenure

    Notes
    -----
    This pipeline modifies the input DataFrame in place and returns it.
    """
    # Extract payment automation pattern
    mutate_payment(data)
    # Segment customers into tenure-based lifecycle groups
    mutate_customer_lifetime_buckets(data)
    # Flag newly acquired customers for early churn analysis
    mutate_new_customer_flag(data)
    return data

# ============================================================================
# VALUE & SERVICE RICHNESS FEATURE ENGINEERING PIPELINE
# ============================================================================

def value_x_service_feature_eng(data):
    """
    Create value and service engagement features for customer segmentation.

    Applies a focused feature engineering pipeline targeting customer value
    metrics and service portfolio depth. Ideal for understanding customer
    engagement levels, monetization patterns, and high-value customer identification.

    Parameters
    ----------
    data : pd.DataFrame
        Raw customer data with PaymentMethod, MonthlyCharges, tenure, and
        service columns.

    Returns
    -------
    pd.DataFrame
        Enriched DataFrame with features:
        - automatic_payment: Binary flag for automatic payments
        - value_gap: Pricing discrepancy relative to tenure
        - count_services: Total number of services subscribed (0-9)
        - premium_user_flag: Binary flag for customers with 7+ services

    Notes
    -----
    This pipeline modifies the input DataFrame in place and returns it.
    Focuses on monetization and engagement signals rather than tenure stages.
    """
    # Extract payment automation pattern
    mutate_payment(data)
    # Calculate pricing signals based on tenure and monthly charges
    mutate_value_gap(data)
    # Count service portfolio depth
    mutate_count_services(data)
    # Identify high-value customers based on service subscriptions
    mutate_premium_user_flag(data)
    return data


# ============================================================================
# COMPREHENSIVE FEATURE ENGINEERING PIPELINE
# ============================================================================

def full_feature_eng(data):
    """
    Create comprehensive feature set combining tenure, value, and service metrics.

    Applies the complete feature engineering pipeline, integrating all available
    transformations for maximum feature coverage. Ideal for exploratory analysis
    and comprehensive model training with all available signals.

    Parameters
    ----------
    data : pd.DataFrame
        Raw customer data with all required columns: PaymentMethod, tenure,
        MonthlyCharges, and service subscription columns.

    Returns
    -------
    pd.DataFrame
        Enriched DataFrame with all engineered features:
        - automatic_payment: Binary flag for automatic payments
        - customer_lifetime_buckets: Tenure segmentation (0-5, 5-10, 10-20, 20-40, 60+)
        - new_customer_flag: Binary flag for customers with <=6 months tenure
        - value_gap: Pricing discrepancy relative to tenure
        - count_services: Total number of services subscribed (0-9)
        - premium_user_flag: Binary flag for customers with 7+ services

    Notes
    -----
    This pipeline modifies the input DataFrame in place and returns it.
    Combines all three feature engineering pipelines for comprehensive modeling.
    This may result in higher dimensionality but provides all available signals.

    Examples
    --------
    >>> df = pd.read_csv('customer_data.csv')
    >>> df_engineered = full_feature_eng(df)
    """
    # PAYMENT FEATURES
    # Extract payment automation pattern as a binary signal
    mutate_payment(data)

    # TENURE-BASED FEATURES
    # Segment customers into lifecycle stages based on tenure history
    mutate_customer_lifetime_buckets(data)
    # Flag newly acquired customers within first 6 months
    mutate_new_customer_flag(data)

    # VALUE & PRICING FEATURES
    # Calculate pricing signals relative to customer tenure
    mutate_value_gap(data)

    # SERVICE RICHNESS FEATURES
    # Aggregate total service portfolio depth for engagement analysis
    mutate_count_services(data)
    # Identify premium/high-value customers by service count threshold
    mutate_premium_user_flag(data)

    return data
