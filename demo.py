import marimo

__generated_with = "0.15.2"
app = marimo.App(width="medium", layout_file="layouts/demo.slides.json")


@app.cell
def _():
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd

    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import r2_score, root_mean_squared_error

    cleaned_data_df = pd.read_csv('public/data/02-cleaned_posts.csv')
    processed_data_df = pd.read_csv('public/data/03-processed_posts.csv')
    return (
        LinearRegression,
        cleaned_data_df,
        mo,
        pd,
        plt,
        processed_data_df,
        r2_score,
        root_mean_squared_error,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""# Linear regression: LinkedIn post impressions""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Lesson overview

    This lession introduces supervised machine learning using linear regression with a real world problem - predicting the number of impressions a LinkedIn post will recive.

    1. Context
    2. Problem statement
    3. Single linear regression
    4. Multiple linear regression
    5. Summary
    6. Next steps
    7. Miniproject assignment
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.vstack([
        mo.md(
        r"""
        ## 1. Context

        """
        ),
        mo.hstack([
            mo.md(
                r"""
                ### Prerequisites

                1. Python
                2. Pandas
                3. Interpreting plots
                4. Functions (mathematical sense)
                """
            ),
            mo.md(
                r"""
                ### Learning goals

                1. Describe and explain **supervised machine learning**
                2. Understand how **linear regression** models data
                3. **Apply** linear regression to real-world problems
                4. **Evaluate** linear regression model performance
                """
            ),
            mo.md(
                r"""
                ### Tools

                1. **Python**: the glue
                2. **Pandas**: data manipulation
                3. **Scikit-learn**: linear regression
                4. **Matplotlib**: data visualization
                """
            ),
        ], gap=3.0)
    ], gap=2.0)
    return


@app.cell(hide_code=True)
def _(cleaned_data_df, mo):
    mo.vstack([
        mo.md(
        r"""

        ## 2. Problem statement

        **Problem**: The social media team's LinkedIn posts are not getting very many impressions.

        **Solution**: Build a machine learning model which takes information about a post as input and predicts impressions. This will allow us to:

        - Estimate the number of impressions a draft post will get
        - Understand what post characteristics influence impressions

        ### Data
        """
        ),
        mo.hstack(
            [
                mo.md(
                r"""
                1. `impressions`: number of people who saw the post in their feed
                2. `word_count`: words in the post body
                3. `n_tags`: number of tags in the post (ex: #machinelearning)
                4. `external_link`: does the post contain a link an external site or resource?
                5. `media`: was media (image, document, etc) uploaded with the post?
                6. `post_day`: day of the week the post was shared
                """
                ),
                cleaned_data_df.head(10)
            ],
            justify="start",
        )
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    _y_mx_b_fig = 'public/figures/y-mx-b.jpg'
    _model_fig = 'public/figures/wordcount-model-impressions.jpg'

    features = ['word_count', 'n_tags', 'external_link', 'media', 'post_day']

    feature = mo.ui.dropdown(
        features,
        value=None,
        label='Input feature',
    )

    feature_a = mo.ui.dropdown(
        features,
        value=None,
        label='Input feature 1',
    )

    feature_b = mo.ui.dropdown(
        features,
        value=None,
        label='Input feature 2',
    )

    _tex1 = (
        f"$$y = \\beta_0 + \\beta_1 x$$"
    )

    _tex2 = (
        f"$$y = m x + b$$"
    )

    mo.vstack([
        mo.md(r"""
        ## 3. Single linear regression
        A **linear regression** model works by minimizing the difference between it's predictions and the true labels. It does this by adjusting the `beta` parameters in the following equation:
        """),
        mo.hstack([
            mo.vstack([
                mo.md(f"""{_tex1}"""),
                mo.md(r"""
                Seem familiar? You may have seen it written this way in highschool algebra:
                """),
                mo.md(f"""{_tex2}"""),
            ]),
            mo.center(mo.image(src=_y_mx_b_fig)),
        ]),
        mo.md(r"""
        It's the equation of a line! Here `m` is the slope of the line, and `b` is the intercept - where the line crosses the y-axis. We can use linear regression to predict how many impressions a post will get (y) based on one of our input features (x). """),
        mo.center(mo.image(src=_model_fig, width='500px')),
        mo.md(r"""
        Linear regression is a **supervised machine learning** model. We train it by providing known x and y pairs. The model 'learns' by finding the best slope and intercept. It does this by calculating y for all input x values and then comparing the calculated y to the true y. The best slope and intercept give the smallest difference between true and calculated y values.
        """),
    ])
    return feature, feature_a, feature_b


@app.cell
def _(
    LinearRegression,
    feature,
    mo,
    pd,
    plt,
    processed_data_df,
    r2_score,
    root_mean_squared_error,
):
    def plot_single_regression(processed_data_df, single_linear_model, feature):

        fig, axs = plt.subplots(1, 3, figsize=(12, 3.5))

        fig.suptitle(f'Single linear regression model performance\nR\u00b2 = 0.000, RMSE = 000')

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

            result = single_linear_model.fit(
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

            fig.suptitle(f'Single linear regression model performance\nR\u00b2 = {rsq:.3f}, RMSE = {rmse:.0f}')

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


    single_linear_model = LinearRegression()

    single_regression_plot, coef, intercept = plot_single_regression(
        processed_data_df,
        single_linear_model,
        feature.value
    )

    if feature.value is not None:
        feature_name = feature.value.replace('_', '\_')

    _tex = (
        f"$$y = \\beta_1 x + \\beta_0$$" if coef is None else f"$$y = {coef[0]:.1f} ({feature_name}) + {int(intercept)}$$"
    )

    mo.vstack([
        mo.md(r"""
        ## 3. Single linear regression

         Choose a feature see how a linear regression model using it as the only input performs.
        """),
        feature,
        mo.as_html(single_regression_plot),
        mo.md(f"""{_tex}"""),
        mo.md(r"""
        ### Model evaluation

        - **R<sup>2</sup>**: This metric tells you what fraction of the variation in impressions is explained by the model, larger is better - up to 1.0.
        - **RMSE**: This metric tells you on average, how many impressions the model is off by, smaller is better.
        """),
    ])
    return


@app.cell
def _(
    LinearRegression,
    feature_a,
    feature_b,
    mo,
    plt,
    processed_data_df,
    r2_score,
    root_mean_squared_error,
):
    def plot_multiple_regression(processed_data_df, multiple_linear_model, feature_a, feature_b):

        fig, axs = plt.subplots(1, 2, figsize=(12, 4.5))

        fig.suptitle(f'Multiple linear regression model performance\nR\u00b2 = 0.000, RMSE = 000')

        axs[0].set_title('Model predictions')
        axs[0].set_xlabel('True impressions')
        axs[0].set_ylabel('Predicted impressions')

        axs[1].set_title('Fit residuals')
        axs[1].set_xlabel('Predicted impressions')
        axs[1].set_ylabel('True - predicted impressions')

        if feature_a is not None and feature_b is not None:

            model_features = [feature_a, feature_b]

            result = multiple_linear_model.fit(
                processed_data_df[model_features],
                processed_data_df['impressions']
            )

            coef = multiple_linear_model.coef_
            intercept = multiple_linear_model.intercept_

            feature_values = processed_data_df[model_features]
            labels = processed_data_df['impressions']
            predictions = multiple_linear_model.predict(processed_data_df[model_features])

            residuals = labels - predictions

            rsq = r2_score(labels, predictions)
            rmse = root_mean_squared_error(labels, predictions)

            fig.suptitle(f'Multiple linear regression model performance\nR\u00b2 = {rsq:.3f}, RMSE = {rmse:.0f}')

            axs[0].scatter(labels, predictions, color='black')
            axs[0].axline((0, 0), slope=1, color='red', linestyle='--', label='Ideal fit')
            axs[0].legend(loc='best')

            axs[1].scatter(predictions, residuals, color='black')
            axs[1].axhline(0, color='red', linestyle='--', label='Ideal fit')
            axs[1].legend(loc='best')

        fig.tight_layout()

        return fig, multiple_linear_model

    _tex1 = (
        f"$$y = \\beta_0 + \\beta_1 x_1 + \\beta_2 x_2$$"
    )

    multiple_linear_model = LinearRegression()

    multiple_regression_plot, multiple_linear_model = plot_multiple_regression(
        processed_data_df,
        multiple_linear_model,
        feature_a.value,
        feature_b.value
    )

    mo.vstack([
        mo.md(r"""
        ## 4. Multiple linear regression
        We can probably improve our single linear regression model by adding more features. For two input features the equation looks like this:
        """),
        mo.md(f"""{_tex1}"""),
        mo.hstack([feature_a, feature_b], justify='start'),
        mo.as_html(multiple_regression_plot),
        mo.md(r"""
        **Questions**:

        - Is the new model better? If so, by how much? Is it a lot better or only a little?
        - What are some weaknesses of the model? How could we make it better?
        """),
    ])
    return (multiple_linear_model,)


@app.cell(hide_code=True)
def _(
    feature_a,
    feature_b,
    mo,
    multiple_linear_model,
    pd,
    plt,
    processed_data_df,
):
    def plot_feature_importance(multiple_linear_model, processed_data_df, feature_a, feature_b):

        fig, ax = plt.subplots(1,1, figsize=(6,4))

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

    importance_plot = plot_feature_importance(
        multiple_linear_model,
        processed_data_df,
        feature_a.value,
        feature_b.value
    )

    mo.vstack([
        mo.md(
        r"""
        ## 4. Multiple linear regression

        A fitted linear regression model is **interpretable** i.e., it can tell us something about the relative impact of the features on the label.
        """
        ),
        mo.vstack([mo.as_html(importance_plot)], heights=[0.75], align='center'),
        mo.md(r"""
        The regression coefficents (values of `beta`) describe the relative strength and direction of each features relationship with the label. Here, posts with more tags tend to get more impressions. Longer posts also tend to get more impressions, but the effect is not quite as strong.
        """
        )
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## 5. Summary

    1. **Supervised machine learning** uses labeled data to construct a model that can predict outputs from inputs
    2. **Linear regression** models data as a linear combination of input features multiplied by coefficients
    3. **Single linear regression** uses one input feature, while **multiple linear regression** uses two or more features
    4. **R<sup>2</sup>** is a metric that reports the fraction of variation in the label that is described by the model, higher is better
    5. **RMSE** (root mean squared error) is a metric that reports, on average, how much the model is off by in its predictions
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## 6. Next steps

    Future lessons will include:

    - Preprocessing and preparing data for modeling
    - More advanced techniques for model evaluation
    - More powerful model types
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## 7. Miniproject assignment

    See the notebook `post_impressions_assignment.ipynb` in the notebooks folder of this repository. The full solution is in the `post_impressions_full_solution.ipynb` notebook, feel free to take a look if you get stuck. But be aware - there are some advanced techniques used in the solution that may make things seem more complicated, not less.
    """
    )
    return


if __name__ == "__main__":
    app.run()
