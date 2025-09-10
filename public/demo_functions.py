"""
Helper module to extract demo functions from demo.py for testing.
This module provides standalone versions of the plotting functions without the marimo app context.
"""

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import r2_score, root_mean_squared_error


def plot_single_regression(processed_data_df, single_linear_model, feature):
    """
    Create a single linear regression visualization with three subplots.
    
    Args:
        processed_data_df (pd.DataFrame): DataFrame containing the data
        single_linear_model: Sklearn LinearRegression model
        feature (str): Name of the feature column to use
        
    Returns:
        tuple: (figure, coefficient, intercept)
    """
    fig, axs = plt.subplots(1, 3, figsize=(12, 3.5))

    fig.suptitle(f'Single linear regression model performance\nR² = 0.000, RMSE = 000')

    axs[0].set_title(f'Training data')
    axs[0].set_xlabel(f'Feature')
    axs[0].set_ylabel('Impressions')

    axs[1].set_title('Model predictions')
    axs[1].set_xlabel('True impressions')
    axs[1].set_ylabel('Predicted impressions')

    axs[2].set_title('Fit residuals')
    axs[2].set_xlabel('Predicted impressions')
    axs[2].set_ylabel('True - predicted impressions')

    coef = None
    intercept = None

    if feature is not None:

        _ = single_linear_model.fit(
            processed_data_df[feature].to_frame(),
            processed_data_df['impressions']
        )

        coef = single_linear_model.coef_
        intercept = single_linear_model.intercept_

        feature_values = processed_data_df[feature]
        labels = processed_data_df['impressions']
        predictions = single_linear_model.predict(processed_data_df[feature].to_frame())

        predictions_df = pd.DataFrame({
            feature: feature_values,
            'labels': labels,
            'predictions': predictions
        })

        predictions_df.sort_values(by='predictions', inplace=True)

        residuals = labels - predictions

        rsq = r2_score(labels, predictions)
        rmse = root_mean_squared_error(labels, predictions)

        fig.suptitle(f'Single linear regression model performance\nR² = {rsq:.3f}, RMSE = {rmse:.0f}')

        axs[0].scatter(processed_data_df[feature], labels, color='black')
        axs[0].plot(predictions_df[feature], predictions_df['predictions'], color='red', label='Model')
        axs[0].set_xlabel(f'{feature}')
        axs[0].set_ylabel('True impressions')
        axs[0].legend(loc='best')

        axs[1].scatter(labels, predictions, color='black')
        axs[1].axline((0, 0), slope=1, color='red', linestyle='--', label='Ideal fit')
        axs[1].legend(loc='best')

        axs[2].scatter(predictions, residuals, color='black')
        axs[2].axhline(0, color='red', linestyle='--', label='Ideal fit')
        axs[2].legend(loc='best')

    fig.tight_layout()

    return fig, coef, intercept


def plot_multiple_regression(processed_data_df, multiple_linear_model, feature_a, feature_b):
    """
    Create a multiple linear regression visualization with two subplots.
    
    Args:
        processed_data_df (pd.DataFrame): DataFrame containing the data
        multiple_linear_model: Sklearn LinearRegression model
        feature_a (str): Name of the first feature column
        feature_b (str): Name of the second feature column
        
    Returns:
        tuple: (figure, fitted_model)
    """
    fig, axs = plt.subplots(1, 2, figsize=(12, 4.5))

    fig.suptitle(f'Multiple linear regression model performance\nR² = 0.000, RMSE = 000')

    axs[0].set_title('Model predictions')
    axs[0].set_xlabel('True impressions')
    axs[0].set_ylabel('Predicted impressions')

    axs[1].set_title('Fit residuals')
    axs[1].set_xlabel('Predicted impressions')
    axs[1].set_ylabel('True - predicted impressions')

    if feature_a is not None and feature_b is not None:

        model_features = [feature_a, feature_b]

        _ = multiple_linear_model.fit(
            processed_data_df[model_features],
            processed_data_df['impressions']
        )

        labels = processed_data_df['impressions']
        predictions = multiple_linear_model.predict(processed_data_df[model_features])

        residuals = labels - predictions

        rsq = r2_score(labels, predictions)
        rmse = root_mean_squared_error(labels, predictions)

        fig.suptitle(f'Multiple linear regression model performance\nR² = {rsq:.3f}, RMSE = {rmse:.0f}')

        axs[0].scatter(labels, predictions, color='black')
        axs[0].axline((0, 0), slope=1, color='red', linestyle='--', label='Ideal fit')
        axs[0].legend(loc='best')

        axs[1].scatter(predictions, residuals, color='black')
        axs[1].axhline(0, color='red', linestyle='--', label='Ideal fit')
        axs[1].legend(loc='best')

    fig.tight_layout()

    return fig, multiple_linear_model


def plot_feature_importance(multiple_linear_model, feature_a, feature_b):
    """
    Create a feature importance visualization showing regression coefficients.
    
    Args:
        multiple_linear_model: Fitted sklearn LinearRegression model
        feature_a (str): Name of the first feature
        feature_b (str): Name of the second feature
        
    Returns:
        matplotlib.figure.Figure: The created figure
    """
    fig, ax = plt.subplots(1, 1, figsize=(6, 4))

    fig.suptitle('Feature Importance')
    ax.set_xlabel('Feature')
    ax.set_ylabel('Regression coefficient')

    if feature_a is not None and feature_b is not None:
        importance = pd.DataFrame({
            'Feature': [feature_a, feature_b],
            'Importance': multiple_linear_model.coef_
        })

        importance.sort_values(by='Importance', inplace=True)

        ax.bar(importance['Feature'], importance['Importance'], color='black')

    fig.tight_layout()

    return fig
