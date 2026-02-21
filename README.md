# Sales Data Analysis Using Linear Regression

A comprehensive sales forecasting project that uses linear algebra and machine learning techniques to analyze retail sales data and predict future sales performance across different regions in Germany.

## Project Overview

This project analyzes synthetic sales data from a retail company operating across five regions in Germany. Using linear regression and exploratory data analysis, we identify key trends, patterns, and factors that influence sales performance.

## Features

- **Synthetic Dataset Generation**: Creates realistic sales data for 2 years (730 days)
- **Exploratory Data Analysis**: Visualizes sales trends, correlations, and patterns
- **Linear Regression Model**: Predicts sales based on multiple features
- **Performance Metrics**: Evaluates model accuracy using MAE, RMSE, and R² score
- **Regional Analysis**: Compares sales performance across German regions
- **Feature Importance**: Identifies key drivers of sales

## Technologies Used

- **Python 3.x**
- **pandas** - Data manipulation and analysis
- **numpy** - Numerical computations
- **scikit-learn** - Machine learning and regression analysis
- **matplotlib** - Data visualization
- **seaborn** - Statistical visualizations

## Installation
# Install required libraries
pip install pandas numpy scikit-learn matplotlib seaborn
```

Or if using Google Colab, run:

```python
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
```

## Dataset Features

The synthetic dataset includes the following features:

| Feature | Description |
|---------|-------------|
| **Date** | Daily records from 2022-01-01 to 2023-12-31 |
| **Sales_Amount** | Total revenue in Euros (€) |
| **Products_Sold** | Number of products sold per day |
| **Marketing_Expenditure** | Daily marketing spend in Euros (€) |
| **Region** | Five German regions: Bavaria, Baden-Württemberg, North Rhine-Westphalia, Hesse, Saxony |

## Usage

### 1. Generate Dataset
```python
# Run the dataset generation code to create 730 days of sales data
# Data includes realistic correlations between features
```

### 2. Exploratory Data Analysis
```python
# Visualize sales trends over time
# Analyze correlation between features
# Compare regional performance
```

### 3. Train Linear Regression Model
```python
# Split data into 80% training, 20% testing
# Train model on historical data
# Make predictions on test set
```

### 4. Evaluate Model Performance
```python
# Calculate MAE, RMSE, and R² score
# Visualize actual vs predicted sales
# Analyze prediction errors
```

## Key Results

### Model Performance
- **R² Score**: 0.9455 (94.55% variance explained)
- **MAE**: €548.02 (average prediction error)
- **RMSE**: €671.68 (typical prediction error)

### Feature Importance (Coefficients)
1. **Products_Sold**: +€50.38 per product
2. **Marketing_Expenditure**: +€0.77 per euro spent
3. **Region_Hesse**: +€61.80 (best performing region)
4. **Region_North Rhine-Westphalia**: +€27.19
5. **Region_Bavaria**: -€44.42 (underperforming)
6. **Region_Saxony**: -€47.08 (underperforming)

### Business Insights
**Excellent model accuracy** - R² = 94.55%  
**Products sold** is the strongest sales driver  
**Marketing ROI** is positive (€0.77 return per €1 spent)  
**Regional differences** are significant - Hesse outperforms others  
**Bavaria and Saxony** require investigation for underperformance

## Visualizations

The project includes the following visualizations:

1. **Sales Trend Over Time** - Line chart showing daily sales patterns
2. **Marketing vs Sales** - Scatter plot revealing correlation
3. **Regional Performance** - Bar chart comparing average sales by region
4. **Products Sold Distribution** - Histogram of product sales frequency
5. **Actual vs Predicted Sales** - Model accuracy visualization
6. **Prediction Errors Distribution** - Residual analysis
7. **Feature Importance** - Horizontal bar chart ranking features


## Acknowledgments

- Dataset generated using synthetic data techniques
- Built with scikit-learn machine learning library
- Visualizations created with matplotlib and seaborn
