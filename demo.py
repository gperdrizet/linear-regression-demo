import marimo

__generated_with = "0.15.1"
app = marimo.App(width="medium", layout_file="layouts/demo.slides.json")


@app.cell
def _():
    import marimo as mo
    import pandas as pd
    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""# Linear regression: LinkedIn post impressions""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Lesson overview

    1. Context
    2. Problem statement
    3. Demo
    4. Summary
    5. Next steps
    6. Miniproject assignment
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## 1. Context

    ### Prerequisites

    Going into this lesson, you should have some understanding of:

    1. Python
    2. Pandas
    3. Interpreting plots
    4. Functions (mathematical sense)
    5. LinkedIn/social media

    ### Learning goals

    After completing this module you should be able to:

    1. Describe and explain **supervised machine learning**
    2. Understand how **linear regression** models data
    3. **Apply** linear regression to real-world problems
    4. **Evaluate** linear regression model performance

    ### Tools

    The technologies we will use are:

    1. Python: the glue
    2. Pandas: data manipulation
    3. Scikit-learn: linear regression
    4. Matplotlib: data visualization
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):


    mo.vstack([
        mo.md(
            r"""
        ## 2. The problem
    
        **Problem**: The social media team's LinkedIn posts are not getting very many impressions
    
        **Solution**: Build a machine learning model which takes information about a post as input and predicts impression
    
        ### Data
        """
        ),
        mo.hstack(
            [mo.ui.text(label="hello"), mo.ui.slider(1, 10, label="slider")],
            justify="start",
        )
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## 3. Single linear regression""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## 4. Multiple linear regression""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## 5. Model evaluation""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## 6. Next steps""")
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
