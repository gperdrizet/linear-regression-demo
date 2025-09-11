import unittest
from unittest.mock import patch
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from public.demo_functions import plot_single_regression, plot_multiple_regression, plot_feature_importance

class TestPlottingFunctions(unittest.TestCase):
    """Test cases for plotting functions from demo.py"""
    
    def setUp(self):
        """Set up test fixtures before each test method."""

        # Create sample data for testing
        np.random.seed(42)
        self.sample_data = pd.DataFrame({
            'impressions': np.random.randint(100, 1000, 50),
            'word_count': np.random.randint(10, 100, 50),
            'character_count': np.random.randint(50, 500, 50),
            'hashtag_count': np.random.randint(0, 10, 50),
            'mention_count': np.random.randint(0, 5, 50)
        })
        
        # Create a simple linear regression model
        self.model = LinearRegression()
        
    def tearDown(self):
        """Clean up after each test."""
        plt.close('all')
    
    def test_plot_single_regression_creates_figure(self):
        """Test that plot_single_regression creates a figure with correct structure."""

        # Train a simple model
        feature = 'word_count'
        X = self.sample_data[[feature]]
        y = self.sample_data['impressions']
        self.model.fit(X, y)
        
        # Test the plotting function
        with patch('matplotlib.pyplot.show'):
            fig, coef, intercept = plot_single_regression(self.sample_data, self.model, feature)
            
            # Check that a figure was created
            self.assertIsNotNone(fig)
            
            # Check that the figure has 3 subplots
            self.assertEqual(len(fig.axes), 3)
            
            # Check subplot titles
            expected_titles = ['Training data', 'Model predictions', 'Fit residuals']
            
            for i, expected_title in enumerate(expected_titles):
                self.assertIn(expected_title.lower(), fig.axes[i].get_title().lower())
                
            # Check that coefficients and intercept are returned
            self.assertIsNotNone(coef)
            self.assertIsNotNone(intercept)
    
    def test_plot_multiple_regression_creates_figure(self):
        """Test that plot_multiple_regression creates a figure with correct structure."""
        
        # Train a multiple regression model
        feature_a = 'word_count'
        feature_b = 'character_count'
        features = [feature_a, feature_b]
        X = self.sample_data[features]
        y = self.sample_data['impressions']
        self.model.fit(X, y)
        
        # Test the plotting function
        with patch('matplotlib.pyplot.show'):
            fig, fitted_model = plot_multiple_regression(self.sample_data, self.model, feature_a, feature_b)
            
            # Check that a figure was created
            self.assertIsNotNone(fig)
            
            # Check that the figure has 2 subplots
            self.assertEqual(len(fig.axes), 2)
            
            # Check subplot titles
            expected_titles = ['Model predictions', 'Fit residuals']
            
            for i, expected_title in enumerate(expected_titles):
                self.assertIn(expected_title.lower(), fig.axes[i].get_title().lower())
                
            # Check that the fitted model is returned
            self.assertIsNotNone(fitted_model)
    
    def test_plot_feature_importance_creates_figure(self):
        """Test that plot_feature_importance creates a figure with correct structure."""
        
        # Train a multiple regression model
        feature_a = 'word_count'
        feature_b = 'character_count'
        features = [feature_a, feature_b]
        X = self.sample_data[features]
        y = self.sample_data['impressions']
        self.model.fit(X, y)
        
        # Test the plotting function
        with patch('matplotlib.pyplot.show'):
            fig = plot_feature_importance(self.model, feature_a, feature_b)
            
            # Check that a figure was created
            self.assertIsNotNone(fig)
            
            # Check that the figure has 1 subplot
            self.assertEqual(len(fig.axes), 1)
            
            # Check the title
            self.assertIn('feature importance', fig._suptitle.get_text().lower())
            
            # Check axis labels
            ax = fig.axes[0]
            self.assertIn('feature', ax.get_xlabel().lower())
            self.assertIn('coefficient', ax.get_ylabel().lower())
    
    def test_plot_single_regression_with_valid_data(self):
        """Test plot_single_regression with valid input data."""
        
        feature = 'word_count'
        X = self.sample_data[[feature]]
        y = self.sample_data['impressions']
        self.model.fit(X, y)
        
        # Test that the function doesn't raise an exception
        try:
            with patch('matplotlib.pyplot.show'):
                fig, coef, intercept = plot_single_regression(self.sample_data, self.model, feature)
                
                # Check that we get a matplotlib figure
                self.assertIsInstance(fig, plt.Figure)
        
        except Exception as e:
            self.fail(f"plot_single_regression raised an exception: {e}")
    
    def test_plot_multiple_regression_with_valid_data(self):
        """Test plot_multiple_regression with valid input data."""
        
        feature_a = 'word_count'
        feature_b = 'character_count'
        features = [feature_a, feature_b]
        X = self.sample_data[features]
        y = self.sample_data['impressions']
        self.model.fit(X, y)
        
        # Test that the function doesn't raise an exception
        try:
            with patch('matplotlib.pyplot.show'):
                fig, fitted_model = plot_multiple_regression(self.sample_data, self.model, feature_a, feature_b)
                
                # Check that we get a matplotlib figure
                self.assertIsInstance(fig, plt.Figure)
        
        except Exception as e:
            self.fail(f"plot_multiple_regression raised an exception: {e}")
    
    def test_plot_feature_importance_with_valid_data(self):
        """Test plot_feature_importance with valid input data."""
        
        feature_a = 'word_count'
        feature_b = 'character_count'
        features = [feature_a, feature_b]
        X = self.sample_data[features]
        y = self.sample_data['impressions']
        self.model.fit(X, y)
        
        # Test that the function doesn't raise an exception
        try:
            with patch('matplotlib.pyplot.show'):
                fig = plot_feature_importance(self.model, feature_a, feature_b)
                # Check that we get a matplotlib figure
                self.assertIsInstance(fig, plt.Figure)
        except Exception as e:
            self.fail(f"plot_feature_importance raised an exception: {e}")
    
    def test_plot_functions_handle_none_features(self):
        """Test that plotting functions handle None feature values gracefully."""
        
        # Test multiple regression with None features
        with patch('matplotlib.pyplot.show'):
            try:
                fig, fitted_model = plot_multiple_regression(self.sample_data, self.model, None, None)
                self.assertIsInstance(fig, plt.Figure)
            except Exception as e:
                self.fail(f"plot_multiple_regression with None features raised an exception: {e}")
        
        # Test feature importance with None features
        with patch('matplotlib.pyplot.show'):
            try:
                fig = plot_feature_importance(self.model, None, None)
                self.assertIsInstance(fig, plt.Figure)
            except Exception as e:
                self.fail(f"plot_feature_importance with None features raised an exception: {e}")


if __name__ == '__main__':
    unittest.main()
