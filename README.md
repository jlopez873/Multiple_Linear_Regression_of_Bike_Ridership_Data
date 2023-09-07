# Multiple Linear Regression of Bike Ridership Data

This repository contains the code and data for a multiple linear regression analysis of Citi Bike ridership data in New York City. The analysis aims to predict Citi Bike ridership demand using a comprehensive data-driven approach.

## Directory Structure

The repository is organized as follows:

- **clean_data**: Contains cleaned and processed data in Parquet format.
  - bike_stations.parquet
  - clean_weather.parquet
  - compiled_data.parquet
  - engineered_data.parquet
  - holidays.parquet
  - hourly_tripdata.parquet
  - nyc_bike_counts.parquet
  - transformed_data.parquet

- **notebooks**: Jupyter notebooks used for data analysis and modeling.
  - nb1_data_gathering.ipynb
  - nb2_data_cleaning.ipynb
  - nb3_data_aggregation.ipynb
  - nb4_feature_engineering.ipynb
  - nb5_data_transformation.ipynb
  - nb6_data_exploration.ipynb
  - nb7_time_series_analysis.ipynb
  - nb8_model_development.ipynb

- **raw_data**: Contains the raw data used in the analysis.
  - Bicycle_Counters.csv
  - New York City, USA 2018-06-01 to 2023-05-31.csv

- **scripts**: Python scripts used for data gathering and model functions.
  - data_gathering.py
  - model_functions.py

- **Executive Summary - Multiple Linear Regression Analysis of Citi Bike Ridership Data.pdf**: The executive summary of the analysis.

- **Presentation - Multiple Linear Regression Analysis of Citi Bike Ridership Data.pptx**: A presentation summarizing key findings and insights.

- **Report - Multiple Linear Regression Analysis of Citi Bike Ridership Data.pdf**: A detailed report of the analysis, including methodology, key findings, limitations, proposed actions, and expected benefits.

## Executive Summary

The executive summary provides an overview of the analysis, including the problem statement, hypothesis, methodology, key findings, limitations, proposed actions, and expected benefits. It was prepared by Javier Lopez, the Data Analyst.

You can access the full executive summary, presentation, and report for a comprehensive understanding of the analysis and its outcomes.

## Key Findings

The analysis uncovered several key findings related to Citi Bike ridership patterns, influential variables, and the performance of predictive models. Some of the key findings include:

- Weekly, monthly, and yearly ridership trends.
- Seasonal distribution of ridership.
- Correlation analysis of weather variables.
- Influence of holidays and the COVID-19 pandemic on ridership.
- Impact of rider types and pandemic periods.

The full details of these findings are available in the report.

## Limitations and Proposed Actions

The analysis acknowledges certain limitations, such as the challenges of log-transformation, limitations of autocorrelation plots, and the need for more advanced feature selection methods. To address these limitations, several proposed actions are outlined in the report, aiming to refine the analysis process and enhance predictive modeling.

## Expected Benefits

The analysis anticipates significant benefits for Citi Bike operations and management, including improved prediction accuracy, optimized resource allocation, enhanced revenue projection, targeted marketing strategies, cost savings, and improved user satisfaction. The extent of these benefits depends on the accuracy of the predictive models and the implementation of proposed actions.

For a detailed understanding of the analysis and its outcomes, please refer to the full executive summary, presentation, and report in this repository.
