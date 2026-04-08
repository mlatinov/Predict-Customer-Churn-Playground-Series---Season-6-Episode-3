
from sklearn.pipeline import Pipeline
import dalex as dx 
import mlflow
import tempfile
import os
import matplotlib.pyplot as plt

def dx_transform(pipeline, x_train, y_train) : 
    """
    Prepare and transform data for use with a DALEX Explainer.

    Strips the model estimator from the pipeline and applies only the
    preprocessing steps to produce the transformed feature matrix expected
    by DALEX. Also encodes the target variable from string ('Yes'/'No') to
    integer (1/0) if necessary.
    """
    # Clean the target before the preproc to avoid erros lates in Dalex 
    if y_train.dtype == object:
        y_train = (y_train == "Yes").astype(int)

    # Split at the model boundary
    transformer_steps = Pipeline(pipeline.steps[: -1])
    model_only = pipeline.steps[-1][1]

    # Transform the data 
    X_transformed = transformer_steps.transform(x_train)

    results = {
        "model"    : model_only,
        "dalex_df" : X_transformed,
        "y_encoded" : y_train
    }
    return results


def dx_create_explainer(pipeline, x_train, y_train, label) :
    """
    Create a DALEX Explainer from a fitted sklearn Pipeline.

    Extracts the model and preprocessed data from the pipeline via
    dx_transform(), then instantiates a dalex.
    """
    # Transform the data from the pipeline 
    dx_transformed = dx_transform(
        pipeline = pipeline,
        x_train = x_train,
        y_train = y_train
    )
    # Create Dalex Explainer 
    dx_explainer = dx.Explainer(
        model = dx_transformed["model"],
        data = dx_transformed["dalex_df"],
        y = dx_transformed["y_encoded"],
        label= label,
        verbose= False
    ) 
    return dx_explainer

# ====================== Global Feature Importance ========================================
def dx_loss_shuffle_imp(dalex_explainer):
    """
    Function to Compute and Return Shuffle Loss Model Features Importance
    """
    # Compute importance
    imp = dalex_explainer.model_parts(loss_function="1-auc")

    # Extract results
    key_features = imp.result

    # Remove baseline BEFORE sorting
    key_features = key_features[key_features["variable"] != "_baseline_"]

    top_5 = (
        key_features
        .sort_values(by="dropout_loss", ascending=False)
        .head(5)
        .variable
    )
    results = {
        "Top_5_Features": top_5,
        "Loss_Shuffle_Importance_cv": key_features,
        "Importance": imp
    }
    return results

def dx_profiles(dalex_explainer, features_names, type = "partial", groups = None ) :
    """
    Compute Partial Dependency Profiles (PDP) or Accumulated Local Effects (ALE).
    Runs DALEX model_profile on the specified features using a sample of 500
    observations. 
    """
    # Compute the Partial Dependency Profile / ALE 
    profile = dalex_explainer.model_profile(
        type= type,
        variables= features_names,
        N = 500,
        groups= groups
    )
    # Take the results CV
    results_cv = profile.result

    results = {
        "PDP_cv"   : results_cv,
        "PDP_plot" : profile
    }
    return results

def dx_rcdr(dalex_explainer) :
    """
    Compute the Reverse Cumulative Distribution of Residuals (RCDR)
    """
    # Compute the Global residuals 
    mp = dalex_explainer.model_performance()

    return mp

def dx_residual_analysis(dx_explainer,x_transformed, y_encoded) :
    """
    Function to Conpute the and produce plots about the Residuals Distribution from the model 
    """
    # Take the model performance 
    mp      = dx_explainer.model_performance()
    label   = mp.residuals["label"].iloc[0]
    results = mp.residuals.copy()

    # Segment by true class
    churned     = results[results["y"] == 1]
    not_churned = results[results["y"] == 0]

    # Plot the histogram 
    fig1, ax = plt.subplots(figsize=(9, 4))
    ax.hist(not_churned["y_hat"], bins=50, alpha=0.6,
            label="True No  (0)", color="#378ADD", density=True)
    ax.hist(churned["y_hat"],     bins=50, alpha=0.6,
            label="True Yes (1)", color="#D85A30", density=True)
    ax.axvline(0.5, color="black", linestyle="--", linewidth=1, label="threshold 0.5")
    ax.set_xlabel("Predicted probability")
    ax.set_ylabel("Density")
    ax.set_title(f"Score distribution by true class — {label}")
    ax.legend()
    plt.tight_layout()

def dx_global_importance(dalex_explainer, features_names, groups = None) :
    """
    Compute a comprehensive set of global model explanations.

    Combines three complementary global explanation techniques:
    - Permutation-based (Shuffle Loss) feature importance using 1-AUC as the loss function.
    - Partial Dependency Profiles (PDP) for the specified features.
    - Reverse Cumulative Distribution of Residuals (RCDR) for model performance
      diagnostics.
    """
    # Run Shuffle Importance with 1 - AUC as a loss function 
    features_imp = dx_loss_shuffle_imp(dalex_explainer)

    # Run PDP or ALE 
    partial_profiles = dx_profiles(dalex_explainer, features_names)

    # Run Reverse Cumulative Distribution of Residuals 
    rcdr = dx_rcdr(dalex_explainer)

    results = {
        "Top_5_Features": features_imp["Top_5_Features"],
        "Loss_Shuffle_Importance_cv": features_imp["Loss_Shuffle_Importance_cv"],
        "Loss_Shuffle_Plot": features_imp["Importance"],
        "PDP_Plot" : partial_profiles["PDP_plot"],
        "RCDR_Plot" : rcdr

    }
    return results

# ======================  INSTANCE LEVEL EXPLANATIONS =======================
def dx_build_profiles(pipeline, x_train, y_train) :
    """
    Build transformed observation profiles for local explanation analysis.

    Randomly samples one churned customer (Churn='Yes') and one non-churned
    customer (Churn='No') from the training set, then applies the pipeline's
    preprocessing steps to produce DALEX-compatible feature vectors.
    """
    # Take a Random Sample 
    profile_1_raw = x_train[y_train == "Yes"].sample(n = 1,random_state = 42)
    profile_2_raw = x_train[y_train == "No"].sample(n = 1,random_state = 42)
    # Transform the Sample with the pipeline 
    profile_1 = dx_transform(pipeline = pipeline, x_train = profile_1_raw, y_train = y_train)["dalex_df"]
    profile_2 = dx_transform(pipeline = pipeline, x_train = profile_2_raw, y_train = y_train)["dalex_df"]

    profiles = {
        "profile_yes" : profile_1,
        "profile_no"  : profile_2
    }
    return profiles

def dx_bd(dalex_explainer, profiles) :
    """
    Run Break Down attribution analysis on churned and non-churned profiles.
    """
    # Break down the profiles 
    bd_profile_1 = dalex_explainer.predict_parts(new_observation= profiles["profile_yes"],type = "break_down")
    bd_profile_2 = dalex_explainer.predict_parts(new_observation= profiles["profile_no"],type = "break_down")
    results = {
        "bd_plot_1" : bd_profile_1,
        "bd_plot_2" : bd_profile_2
    }
    return results

def dx_cp(dalex_explainer, profiles, features) :
    """
    Run Ceteris Paribus (what-if) analysis on churned and non-churned profiles.
    """
    # Ceratus Paribus (What if analysis)
    cp_profile_1 = dalex_explainer.predict_profile(
        new_observation=profiles["profile_yes"],
        type="ceteris_paribus",
        variables = features
    )
    cp_profile_2 = dalex_explainer.predict_profile(
        new_observation=profiles["profile_yes"],
        type="ceteris_paribus",
        variables = features
    )
    results = {
        "cp_plot_1" : cp_profile_1,
        "cp_plot_2" : cp_profile_2
    }
    return results

def dx_local_explanations(dalex_explainer, pipeline, features_names, x_train, y_train) :
    """
    Generate instance-level explanations for representative churned and non-churned customers.

    Orchestrates the full local explanation workflow: samples representative
    observations via dx_build_profiles, computes Break Down attributions via
    dx_bd, and computes Ceteris Paribus profiles via dx_cp for both the churned
    and non-churned observations.
    """
    # Build Profiles from the data 
    profiles = dx_build_profiles(pipeline, x_train, y_train)

    # Run Break Down Analysis 
    bd = dx_bd(dalex_explainer, profiles)

    # Run Ceretis Paribus Analysis 
    cp = dx_cp(dalex_explainer, profiles, features = features_names)

    results = {
        "bd_plot_1" : bd["bd_plot_1"],
        "bd_plot_2" : bd["bd_plot_2"],
        "cp_plot_1" : cp["cp_plot_1"],
        "cp_plot_2" : cp["cp_plot_2"]
    }
    return results

def mlflow_log_dalex_plot(dalex_result, filename, artifact_path="dalex"):
    """
    Save a DALEX plot as an interactive HTML artifact and log it to MLflow.
    """
    fig = dalex_result.plot(show=False)   

    with tempfile.TemporaryDirectory() as tmp:
        filepath = os.path.join(tmp, filename)
        fig.write_html(filepath)
        mlflow.log_artifact(filepath, artifact_path=artifact_path)