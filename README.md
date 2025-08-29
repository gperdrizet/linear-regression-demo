# Linear Regression Demo: LinkedIn Post Impressions

[![Codespaces Prebuilds](https://github.com/gperdrizet/linear-regression-demo/actions/workflows/codespaces/create_codespaces_prebuilds/badge.svg)](https://github.com/gperdrizet/linear-regression-demo/actions/workflows/codespaces/create_codespaces_prebuilds)

A hands-on data science lesson teaching linear regression concepts through real-world LinkedIn post impression data. This repository contains interactive demonstrations and assignments designed for introductory data science bootcamp students.

## Lesson Overview

This lesson teaches fundamental linear regression concepts using LinkedIn post impression data as a practical, relatable example. Students will learn:

- **Single linear regression**: Understanding relationships between one feature and post impressions
- **Multiple linear regression**: Building models with multiple features for better predictions
- **Model evaluation**: Using R² scores and visualizations to assess model performance
- **Real-world application**: Working with actual social media engagement data

### Learning Objectives

By completing this lesson, students will be able to:
1. Implement linear regression models using scikit-learn
2. Evaluate model performance using statistical metrics
3. Visualize regression results and interpret findings
4. Compare single vs. multiple feature models
5. Apply regression techniques to real-world datasets

## Getting Started with GitHub Codespaces

1. Create a fork of the repository
2. Click the green "Code" button your fork of the repository
3. Select "Codespaces" tab
4. Click "Create codespace on main"
5. Wait for the environment to set up automatically (2-3 minutes)

## How to Use This Repository

### 1. Interactive Demo (`demo.py`)
The main teaching tool is an interactive Marimo notebook that automatically launches when you start a Codespace.

- **Access**: Opens automatically in your browser when Codespace starts
- **Content**: Complete lesson with explanations, code examples, and visualizations
- **Features**: Interactive widgets, step-by-step progression, embedded exercises

### 2. Assignment Notebook
Students work through hands-on exercises in the Jupyter notebook:

**Assignment Tasks**:
- Load and explore the LinkedIn post dataset
- Implement single linear regression
- Build and evaluate multiple linear regression models
- Compare model performance using R² scores
- Create visualizations of results

### 3. Solution Notebook
Instructors can reference the complete solution notebook.


## Project Structure

```
linear-regression-demo/
├── README.md                                # This file
├── demo.py                                  # Interactive Marimo lesson demo
├── requirements.txt                         # Python dependencies
├── LICENSE                                  # GNU GPL license
├── .devcontainer                            # GitHub Codespace configuration
│
├── notebooks/
│   ├── post_impressions_assignment.ipynb    # Student assignment
│   └── post_impressions_full_solution.ipynb # Complete solution
│
└── public/
    └── data/
        ├── 01-raw_posts.csv                 # Original LinkedIn post data
        ├── 02-cleaned_posts.csv             # Cleaned dataset
        └── 03-processed_posts.csv           # Preprocessed features for modeling
```

### Dataset Description

The LinkedIn post dataset includes:
- **impressions**: Number of views (target variable)
- **word_count**: Length of post content (standardized)
- **n_tags**: Number of hashtags used (standardized)
- **external_link**: Presence of external links (encoded)
- **media**: Presence of images/videos (encoded)
- **post_day**: Day of week posted (encoded)

## Expected Outcomes

After completing this lesson, students should be able to:

- Build linear regression models
- Understand why multiple features improve prediction accuracy
- Interpret model coefficients in business context
- Create meaningful visualizations of regression results
- Discuss limitations and assumptions of linear regression

## Contributing

This is an educational resource! Contributions welcome:

- **Bug fixes**: Submit issues or pull requests
- **Enhancements**: Suggest additional exercises or datasets
- **Translations**: Help make content accessible in other languages
- **Feedback**: Share experiences using this in your courses
