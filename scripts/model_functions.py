import numpy as np
import pandas as pd

import seaborn as sns
custom_params = {"axes.spines.right": False, "axes.spines.top": False}
sns.set(style="white", rc=custom_params)
sns.set_palette('pastel')
colors = sns.color_palette()

from matplotlib import gridspec
import matplotlib.dates as mdates
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
plt.rcParams['figure.figsize'] = [16,5]
plt.rcParams['figure.facecolor'] = 'cornsilk'
plt.rcParams['axes.labelsize'] = 13
plt.rcParams['axes.titlesize'] = 15
plt.rcParams['axes.titlepad'] = 12
plt.rcParams['axes.labelpad'] = 10

from scipy.stats import anderson
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error as mse, r2_score as r2, mean_absolute_percentage_error as mape
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor as vif
from statsmodels.tsa.arima.model import ARIMA

def train_test_split(X, y, horizon):
    """
    Split the input features and target variable into training and testing sets.

    Parameters:
    X (pandas.DataFrame): The input features.
    y (pandas.Series): The target variable.
    test_days (int): The test size in days.
    horizon (float): The forecasting horizon in days.

    Returns:
    X_train (pandas.DataFrame): Training input features.
    X_test (pandas.DataFrame): Testing input features.
    y_train (pandas.Series): Training target variable.
    y_test (pandas.Series): Testing target variable.
    """
    window = -round(24 * horizon)

    X_train, X_test = X.iloc[:window], X.iloc[window:]
    y_train, y_test = y.iloc[:window], y.iloc[window:]
    
    return X_train, X_test, y_train, y_test

def scale_data(X_train, X_test, y_train, y_test, ordinal_vars):
    """
    Scale the features and target variables.

    Parameters:
        X_train (pd.DataFrame): Training feature dataset.
        X_test (pd.DataFrame): Testing feature dataset.
        y_train (pd.Series): Training target variable.
        y_test (pd.Series): Testing target variable.
        ordinal_vars (list): List of ordinal variable names.

    Returns:
        X_train_scaled (pd.DataFrame): Scaled training feature dataset.
        X_test_scaled (pd.DataFrame): Scaled testing feature dataset.
        y_train_scaled (pd.Series): Scaled and transformed training target variable.
        y_test_scaled (pd.Series): Scaled and transformed testing target variable.
    """
    # Initialize the scaler
    scaler = StandardScaler()

    # Scale exogenous variables
    for X_df in [X_train, X_test]:
        for col in X_df.columns:
            if col not in ordinal_vars:
                X_df[col] = scaler.fit_transform(X_df[[col]])
                
    # Transform the target
    y_train, y_test = np.log10(y_train), np.log10(y_test)
    
    return X_train, X_test, y_train, y_test

def preprocess_data(X, y, horizon, ordinal_vars):
    """
    Preprocess the data by performing train-test split and scaling.

    Parameters:
        X (pd.DataFrame): Feature dataset.
        y (pd.Series): Target variable.
        horizon (float): Forecasting horizon in days.

    Returns:
        X_train (pd.DataFrame): Scaled training feature dataset.
        X_test (pd.DataFrame): Scaled testing feature dataset.
        y_train (pd.Series): Scaled and transformed training target variable.
        y_test (pd.Series): Scaled and transformed testing target variable.
    """
    X_train, X_test, y_train, y_test = train_test_split(X, y, horizon)
    X_train, X_test, y_train, y_test = scale_data(X_train, X_test, y_train, y_test, ordinal_vars)
    
    return X_train, X_test, y_train, y_test

def get_vif(X):
    """
    Calculate the Variance Inflation Factor (VIF) for each predictor in the dataset.

    Parameters:
        X (pd.DataFrame): Feature dataset.

    Returns:
        vif_data (pd.DataFrame): DataFrame containing variable names and their corresponding VIF values.
    """
    # Create a DataFrame to store the VIF values
    vif_data = pd.DataFrame()
    vif_data["Variable"] = X.columns

    # Calculate the VIF for each predictor
    vif_data["VIF"] = [vif(X.values, i) for i in range(X.shape[1])]
    
    # Sort the DataFrame by VIF values
    vif_data.sort_values('VIF', ascending=False, inplace=True)

    # Round VIF values and reset the index
    vif_data = round(vif_data, 2).reset_index(drop=True)
    
    return vif_data

def get_feature_importance(X_train, y_train):
    """
    Calculate and visualize feature importances using a Random Forest Regressor.

    Parameters:
        X_train (pd.DataFrame): Training features.
        y_train (pd.Series): Target variable for training.

    Returns:
        pd.DataFrame: DataFrame containing feature importances.
    """
    # Create and fit the model
    model = RandomForestRegressor()
    model.fit(X_train, y_train)
    
    # Get feature importances
    importances, features = model.feature_importances_, X_train.columns
    
    # Create a DataFrame with the feature importances
    df = pd.DataFrame({'Feature Name': features, 'Importance': importances}).set_index('Feature Name')
    
    # Sort feature importances in descending order
    df.sort_values('Importance', ascending=False, inplace=True)
    
    # Plot feature importances
    df.plot(kind='bar')
    plt.xticks(rotation=90)
    plt.ylabel('Importance')
    plt.title('Feature Importances', size=17, weight='bold')
    plt.grid(axis='y', alpha=0.5)
    plt.tight_layout()
    plt.show()
    
    return df

def score_model(y_test, y_pred, fit_result=None):
    """
    Calculate evaluation scores for a predictive model.

    Parameters:
        y_test (pd.Series): True target values.
        y_pred (pd.Series): Predicted target values.
        fit_result (ARIMA model): Fitted model.

    Returns:
        scores (dict): Dictionary containing AIC or R2, MAPE and RMSE scores.
    """
    scores = {}
    y_test = 10 ** y_test
    y_pred = 10 ** y_pred
    
    # Calculate AIC and R2 or RMSE
    if fit_result != None:
        scores['AIC'] = fit_result.aic
    else:
        scores['R2'] = r2(y_test, y_pred) * 100
    
    scores['MAPE'] = mape(y_test, y_pred) * 100
    scores['RMSE'] = mse(y_test, y_pred, squared=False)
    scores['Accuracy Rate'] = 100 - scores['MAPE']
    
    return scores

def compile_forecast(y_true, y_pred):
    """
    Compile forecast results including actual, predicted, residuals, and confidence intervals.

    Parameters:
        y_true (pd.Series or np.array): True values.
        y_pred (pd.Series or np.array): Predicted values.

    Returns:
        forecast (pd.DataFrame): DataFrame containing 'Actual', 'Predicted', 'Residuals', 'Lower CI', and 'Upper CI'.
    """
    # Create a DataFrame to store the forecast results
    forecast = pd.DataFrame()
    forecast['Actual'] = y_true
    forecast['Predicted'] = y_pred
    
    # Reverse-transform true and forecast values
    forecast = 10 ** forecast
    
    # Calculate the residuals
    forecast['Residuals'] = forecast['Predicted'] - forecast['Actual']
    
    # Perform bootstrapping for Confidence Intervals
    num_bootstraps = 1000
    bootstrapped_forecasts = []

    for _ in range(num_bootstraps):
        # Randomly sample residuals with replacement
        bootstrap_residuals = np.random.choice(forecast['Residuals'], size=len(forecast), replace=True)

        # Add the sampled residuals to the model's point forecast
        bootstrapped_forecast = forecast['Predicted'] + bootstrap_residuals

        # Append the bootstrapped forecast to the list
        bootstrapped_forecasts.append(bootstrapped_forecast)

    # Calculate the confidence interval
    forecast['Lower CI'] = np.percentile(bootstrapped_forecasts, 2.5, axis=0)
    forecast['Upper CI'] = np.percentile(bootstrapped_forecasts, 97.5, axis=0)
    
    return forecast

def plot_forecast(df, rolling_window):
    """
    Plot actual vs. predicted values along with confidence intervals.

    Parameters:
        df (pd.DataFrame): DataFrame containing 'Actual', 'Predicted', 'Lower CI', and 'Upper CI' columns.
        rolling_window (int): Rolling window size.

    Returns:
        None
    """
    df = df.rolling(24 * rolling_window).mean()
    
    if rolling_window > 1:
        timeframe = f'{rolling_window} Days'
    else:
        timeframe = '24 Hours'
    
    # Plot the actual and predicted lines
    sns.lineplot(data=df[['Actual', 'Predicted']])

    # Fill margin of error and the space between the two lines
    plt.fill_between(df.index, df['Actual'], df['Predicted'], alpha=0.3, color=colors[2])
    plt.fill_between(df.index, df['Lower CI'], df['Upper CI'], alpha=0.15, color=colors[3])
    
    # Add a line at zero
    plt.axhline(y=0, c=colors[3], linestyle='--')

    # Add labels and titles
    plt.xlabel('Date')
    plt.ylabel('Trip Count')
    plt.suptitle('Citi Bike Ridership: Actual vs. Predicted', size='17', weight='bold', y=0.965)
    plt.title(f'(Rolling {timeframe})')

    # Create legend handles
    h1 = mlines.Line2D([], [], linestyle='solid', color=colors[0], label='Actual')
    h2 = mlines.Line2D([], [], linestyle='dashed', color=colors[1], label='Predicted')
    h3 = mpatches.Patch(color=colors[2], alpha=0.3, label='Residual')
    h4 = mpatches.Patch(color=colors[3], alpha=0.15, label='Confidence')
    h5 = mlines.Line2D([], [], linestyle='dashed', color=colors[3], label='No Trips')

    # Add a legend and grid
    plt.legend(handles=[h1, h2, h3, h4, h5], title='Legend')
    plt.grid(axis='x', alpha=0.25)

    # Display the plot
    plt.tight_layout()
    plt.show()
    
def plot_residuals(df):
    """
    Plot various visualizations of residuals.

    Parameters:
        df (pd.DataFrame): DataFrame containing residuals and predicted values.

    Returns:
        None
    """
    # Create a Q-Q plot
    sm.qqplot(data=df['Residuals'], line='s')
    plt.xlabel('Theoretical Quantiles')
    plt.ylabel('Sample Quantiles')
    plt.title('Q-Q Plot of Residuals', size=17, weight='bold')
    plt.tight_layout()
    plt.show()
    
    # Plot the residuals
    sns.scatterplot(data=df['Residuals'])
    plt.axhline(y=0, linestyle='--', c='red')
    plt.xlabel('Date')
    plt.title('Residuals', size=17, weight='bold')
    plt.tight_layout()
    plt.show()

    # Distribution of Residuals
    sns.histplot(data=df['Residuals'], kde=True)
    plt.xlabel('Residuals')
    plt.ylabel('Frequency')
    plt.title('Distribution of Residuals', size=17, weight='bold')
    plt.tight_layout()
    plt.show()

    # Plot residuals vs predicted
    sns.scatterplot(x='Residuals', y='Predicted', data=df)
    plt.xlabel('Residuals')
    plt.ylabel('Predicted')
    plt.title('Residuals vs. Predicted', size=17, weight='bold')
    plt.tight_layout()
    plt.show()
    
def run_arima(params, X_train, X_test, y_train, y_test):
    """
    Run ARIMA model with specified parameters, make forecasts, and calculate scores.

    Parameters:
        params (dict): Dictionary containing 'order' and 'seasonal_order' parameters.
        X_train (pd.DataFrame): Training features.
        X_test (pd.DataFrame): Testing features.
        y_train (pd.Series): Target variable for training.
        y_test (pd.Series): Target variable for testing.

    Returns:
        tuple: ARIMA model, forecasted values, and scores dictionary.
    """
    # Initialize the model
    model = ARIMA(endog=y_train, exog=X_train, order=params['order'], seasonal_order=params['seasonal_order'])

    # Fit the model
    result = model.fit(method='innovations_mle')
    
    # Forecast using the model
    y_pred = result.predict(start=len(y_train), end=len(y_train) + len(y_test) - 1, exog=X_test).values
    
    # Score the model
    scores = score_model(y_test, y_pred, result)

    # Return the results
    return result, y_pred, scores
    
def run_regression(X_train, X_test, y_train, y_test):
    """
    Run Ridge Regression model with hyperparameter tuning, make predictions, and calculate scores.

    Parameters:
        X_train (pd.DataFrame): Training features.
        X_test (pd.DataFrame): Testing features.
        y_train (pd.Series): Target variable for training.
        y_test (pd.Series): Target variable for testing.

    Returns:
        tuple: Ridge Regression model, predicted values, and scores dictionary.
    """
    # Hyperparameter tuning with cross-validation
    param_grid = {'alpha': [0.001, 0.01, 0.1, 1, 10, 100]}  # Test different alpha values
    grid_search = GridSearchCV(estimator=Ridge(), param_grid=param_grid, scoring='neg_mean_squared_error', cv=5)
    grid_search.fit(X_train, y_train)

    # Get the best alpha value
    best_alpha = grid_search.best_params_['alpha']

    # Train the model with the best alpha value
    model = Ridge(alpha=best_alpha)
    result = model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = result.predict(X_test)

    # Calculate R2, MSE, and RMSE
    scores = score_model(y_test, y_pred)

    return result, y_pred, scores
    
def select_features(features, X_train, X_test, y_train, y_test, params=None, model_type=None):
    """
    Select features based on model performance using either ARIMA or regression.

    Parameters:
        features (list): List of features to consider.
        X_train (pd.DataFrame): Training features.
        X_test (pd.DataFrame): Testing features.
        y_train (pd.Series): Target variable for training.
        y_test (pd.Series): Target variable for testing.
        params (dict): Dictionary containing 'order' and 'seasonal_order' parameters.
        model_type (str): Type of model ('arima' or 'regression').

    Returns:
        pd.Dataframe: AIC or R2, RMSE, and features of the best-performing model.
    """
    # Create a DataFrame and list to store model results
    df = pd.DataFrame(columns=[1, 2, 3, 4])
    if model_type == 'arima':
        df.columns = ['AIC', 'MAPE', 'RMSE', 'Accuracy Rate']
    else:
        df.columns = ['R2', 'MAPE', 'RMSE', 'Accuracy Rate']
    features_list = []
    
    # Test model performance with different features
    n = 1
    while n < len(features) + 1:
        # Create and score a model with each set of features
        if model_type == 'arima':
            model, y_pred, scores = run_arima(params, X_train[features[:n]], X_test[features[:n]], y_train, y_test)
        else:
            model, y_pred, scores = run_regression(X_train[features[:n]], X_test[features[:n]], y_train, y_test)
        
        # Add the model's scores and features to df
        df.loc[len(df)] = scores
        features_list.append(features[:n])
        n += 1
        
    # Add the features column and sort by RMSE
    df['features'] = features_list
    df = df.sort_values('Accuracy Rate', ascending=False).reset_index(drop=True)
    
    # Review model performance
    display(df)
    
    # Return the features of the best-performing model
    return df.loc[0]

def run_anderson(df):
    """
    Perform the Anderson-Darling test for normality on the residuals of a model.

    Parameters:
        df (pd.DataFrame): DataFrame containing the residuals.

    Returns:
        None
    """
    # Calculate Anderson-Darling test statistic and critical values
    statistic, critical_values, significance_levels = anderson(df['Residuals'])

    # Print the results
    print("Anderson-Darling Test Statistic:", statistic)

    # Compare the test statistic with critical values at different significance levels
    for i, significance_level in enumerate(significance_levels):
        if statistic > critical_values[i]:
                print(f"{significance_level:.0f}% significance level: Critical Value = {critical_values[i]}, The residuals deviate from normality.")
        else:
            print(f"{significance_level:.0f}% significance level: Critical Value = {critical_values[i]}, The residuals follow a normal distribution.")
            
def find_horizon(X, y, ordinal_vars, params=None, model_type=None):
    """
    Find the best forecast horizon for a given model type.

    Parameters:
        X (pd.DataFrame): Exogenous variables.
        y (pd.Series): Target variable.
        ordinal_vars (list): List of ordinal variable names.
        params (dict): Dictionary containing 'order' and 'seasonal_order' parameters.
        model_type (str): Type of the model ('arima' or 'regression').

    Returns:
        pd.DataFrame: DataFrame containing horizon-specific R2 and RMSE scores.
    """
    # Create a DataFrame to store model scores
    if model_type == 'arima':
        df = pd.DataFrame(columns=['Horizon Months', 'AIC', 'MAPE', 'RMSE', 'Accuracy Rate'])
    else:
        df = pd.DataFrame(columns=['Horizon Months', 'R2', 'MAPE', 'RMSE', 'Accuracy Rate'])
    
    # Find the best forecast horizon
    n = 0
    for i in range(30, 366, 30):
        # Split and preprocess the data based on i
        X_train, X_test, y_train, y_test = preprocess_data(X, y, i, ordinal_vars)

        # Run the model with the new split
        if model_type == 'arima':
            model, y_pred, scores = run_arima(params, X_train, X_test, y_train, y_test)
            df.at[n, ['AIC', 'MAPE', 'RMSE', 'Accuracy Rate']] = scores
        else:
            model, y_pred, scores = run_regression(X_train, X_test, y_train, y_test)
            df.at[n, ['R2', 'MAPE', 'RMSE', 'Accuracy Rate']] = scores

        # Add the results to horizon_df
        df.at[n, 'Horizon Months'] = round(i / 30.04)
        n += 1
        
    df = df.sort_values('Accuracy Rate', ascending=False).reset_index(drop=True)
    
    return df
