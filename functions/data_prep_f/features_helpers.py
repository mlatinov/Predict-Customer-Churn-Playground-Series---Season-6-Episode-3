"""
Feature engineering pipeline orchestration for customer churn prediction.

This module provides high-level pipelines that combine multiple feature
transformations from data_helpers into cohesive feature engineering workflows.
Three pipelines are available with varying levels of feature coverage.
"""
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import PowerTransformer
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import numpy as np 
from functions.data_prep_f.data_helpers import (
    mutate_payment,
    mutate_customer_lifetime_buckets,
    mutate_new_customer_flag,
    mutate_value_gap,
    mutate_count_services,
    mutate_premium_user_flag,
    mutate_model_clean_data
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

    """
    # Extract payment automation pattern
    data = mutate_payment(data)
    # Segment customers into tenure-based lifecycle groups
    data = mutate_customer_lifetime_buckets(data)
    # Flag newly acquired customers for early churn analysis
    data = mutate_new_customer_flag(data)
    # Clean the data for modeling 
    data = mutate_model_clean_data(data)

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

    """
    # Extract payment automation pattern
    data = mutate_payment(data)
    # Calculate pricing signals based on tenure and monthly charges
    data = mutate_value_gap(data)
    # Count service portfolio depth
    data = mutate_count_services(data)
    # Identify high-value customers based on service subscriptions
    data = mutate_premium_user_flag(data)
    # Clean the data for modeling 
    data = mutate_model_clean_data(data)

    return data

# ============================================================================
# COMPREHENSIVE FEATURE ENGINEERING PIPELINE
# ============================================================================

def full_feature_eng(data):
    """
    Create comprehensive feature set combining tenure, value, and service metrics.

    Applies the complete feature engineering pipeline, integrating all available
    transformations for maximum feature coverage. 
    """
    
    # PAYMENT FEATURES
    # Extract payment automation pattern as a binary signal
    data = mutate_payment(data)

    # TENURE-BASED FEATURES
    # Segment customers into lifecycle stages based on tenure history
    data = mutate_customer_lifetime_buckets(data)
    # Flag newly acquired customers within first 6 months
    data = mutate_new_customer_flag(data)

    # VALUE & PRICING FEATURES
    # Calculate pricing signals relative to customer tenure
    data = mutate_value_gap(data)

    # SERVICE RICHNESS FEATURES
    # Aggregate total service portfolio depth for engagement analysis
    data = mutate_count_services(data)
    # Identify premium/high-value customers by service count threshold
    data = mutate_premium_user_flag(data)
     # Clean the data for modeling 
    data = mutate_model_clean_data(data)

    return data

# =============== Sclearn Features Modeling Pipelines ==========================
class FeatureEngineeringTransformer(BaseEstimator, TransformerMixin):

    def __init__(
        self,
        add_value_x_service_feature_eng = True,
        add_tenure_feature_eng = True,
        add_full_feature_eng = True
        ):
        """
        Initialize the FeatureEngineeringTransformer.

        Controls which feature engineering pipelines are applied during the
        transform step. Flags can be combined, but enabling add_full_feature_eng
        already includes all tenure and value-x-service features.
        """
        self.add_value_x_service_feature_eng = add_value_x_service_feature_eng
        self.add_tenure_feature_eng = add_tenure_feature_eng
        self.add_full_feature_eng = add_full_feature_eng
     
    def fit(self, X, y=None):
        self.feature_names_in_ = list(X.columns)
        return self

        # Appliing the transformation funstions 
    def transform(self, X) :
        """
        Apply the configured feature engineering pipelines to the input data.
        Drops identifier columns (customerID, id) and applies the selected
        feature engineering pipelines based on the flags set at initialization.
        If all flags are False, only the base model-clean transformation is applied.
        
        Notes
        -----
        The input DataFrame is copied before transformation to avoid mutating
        the original data. Multiple flags can be True simultaneously, which
        will result in duplicate engineered columns being added.
        """
        X = X.copy()
        X = X.drop(columns=["customerID", "id"], errors="ignore") 

        if self.add_value_x_service_feature_eng :
            X = value_x_service_feature_eng(X)

        if self.add_tenure_feature_eng :
            X = tenure_feature_eng(X)

        if self.add_full_feature_eng :
            X = full_feature_eng(X)

        if not any([
            self.add_value_x_service_feature_eng,
            self.add_tenure_feature_eng,
            self.add_full_feature_eng
        ]):
            X = mutate_model_clean_data(X)
        
        self.feature_names_out_ = list(X.columns)
        return X

        # Method to allow the pipeline to work with Dalex
    def get_feature_names_out(self, input_features=None):
        """
        Return the names of the output features after transformation.
        
        Provides sklearn API compatibility so that downstream steps in a
        Pipeline (e.g., ColumnTransformer) can retrieve feature names
        programmatically. Requires that transform() has been called first.

        """
        return np.array(self.feature_names_out_)

def build_preprocessor(
    add_tenure=True,
    add_value_x_service=True,
    add_full_feature_eng = True,
    add_power_transform = False,
    add_remove_nzv = False
    ) :
    """
    Build an sklearn ColumnTransformer for numerical, categorical, and binary features.
    
    The column lists are expanded dynamically based on which feature engineering pipelines
    are active, so the preprocessor must match the flags passed to FeatureEngineeringTransformer.

    Notes
    -----
    The flags passed here must mirror those passed to FeatureEngineeringTransformer
    to ensure the expected columns are present at preprocessing time.
    """
    # Base Columns that will be present before the feature adding 
    num_columns =  ["tenure","MonthlyCharges","TotalCharges"]
    cat_columns =  [
        "MultipleLines","InternetService","OnlineSecurity","OnlineBackup","DeviceProtection",
        "TechSupport","StreamingTV","StreamingMovies","Contract","PaymentMethod"
    ]
    # Columns that are 0 and 1 encoded before the feature eng 
    binary_columns =  ["Partner","Dependents","PhoneService","PaperlessBilling","gender","SeniorCitizen"]

    # Add the new features if any to the columns for the Column Transformer 
    if add_tenure :
        cat_columns    += ["customer_lifetime_buckets"]
        binary_columns += ["automatic_payment", "new_customer_flag"]

    if add_value_x_service :
        num_columns    += ["value_gap","count_services",]
        binary_columns += ["automatic_payment","premium_user_flag"]

    if add_full_feature_eng :
        cat_columns    += ["customer_lifetime_buckets"]
        num_columns    += ["value_gap", "count_services"]
        binary_columns += ["automatic_payment", "premium_user_flag", "new_customer_flag"]

    # Build a numerical preproc Step by step
    numerical_steps = []

    # Remove near Zero variance Features 
    if add_remove_nzv :
        numerical_steps.append(
            ("nzv", VarianceThreshold(threshold=0.01))
        )

    # Apply Power Transformation Yeo-Johnson
    if add_power_transform :
        numerical_steps.append(
            ("yeo", PowerTransformer(method="yeo-johnson",standardize= False)),
        ) 
    
    # Scale all the numerical Features 
    numerical_steps.append(
        ("scaler", StandardScaler(with_mean=True, with_std=True))
    )
    numerical_transformer = Pipeline(steps=numerical_steps)

    # Build the Preprocessor
    preprocessor = ColumnTransformer(
        transformers= [
            ("numerical", numerical_transformer, num_columns),
            ("cateogorical", OneHotEncoder(handle_unknown="ignore",sparse_output=False) , cat_columns),
            ("binary","passthrough", binary_columns)
        ],
        verbose_feature_names_out=False
    )
    preprocessor.set_output(transform="pandas")
    return preprocessor

def build_pipeline(
    model,
    add_tenure = True,
    add_value_x_service = True,
    add_full_feature = True,
    add_power_transform = False,
    add_remove_nzv = False
    ) :
    """
    Assemble a full sklearn Pipeline combining feature engineering, preprocessing, and a model.

    Creates a three-step Pipeline that sequentially applies:
    1. FeatureEngineeringTransformer – generates domain-specific features.
    2. build_preprocessor – scales numerical, encodes categorical, and passes binary columns.
    3. The supplied estimator – fits or predicts on the preprocessed data.

    Examples
    --------
    >>> from sklearn.linear_model import LogisticRegression
    >>> pipeline = build_pipeline(LogisticRegression(), add_tenure=True, add_value_x_service=False)
    >>> pipeline.fit(X_train, y_train)
    """
    # Pipeline combiners the added features from the Transformer preprocesing from the preprocessor and Model 
    pipeline = Pipeline(steps= [
        ("feature_eng", FeatureEngineeringTransformer(
            add_tenure_feature_eng            = add_tenure,
            add_full_feature_eng              = add_full_feature,
            add_value_x_service_feature_eng   = add_value_x_service
        )),
        ("preprocesing" , build_preprocessor(
            add_tenure           = add_tenure,
            add_value_x_service  = add_value_x_service,
            add_full_feature_eng = add_full_feature,
            add_power_transform  = add_power_transform,
            add_remove_nzv = add_remove_nzv
        )),
        ("model", model)
    ])
    return pipeline


