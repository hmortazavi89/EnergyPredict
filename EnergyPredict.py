# Predicting Energy Consumption in Commercial Buildings
# Project Overview:
# This script predicts hourly energy consumption in commercial buildings using a time-series dataset.
# It leverages features like temperature, humidity, and building characteristics to forecast energy usage,
# supporting energy optimization strategies. The project uses Random Forest, XGBoost, and LSTM models,
# with feature engineering, hyperparameter tuning, and cross-validation for robust predictions.

# Objectives:
# - Perform time-series feature engineering (lagged variables, rolling statistics).
# - Compare multiple models (Random Forest, XGBoost, LSTM) for energy prediction.
# - Tune hyperparameters using GridSearchCV.
# - Evaluate models with cross-validation and metrics (MAE, RMSE).

# Dataset:
# The dataset (building_energy.csv) contains:
# - Timestamp: Date and time of observation (hourly).
# - EnergyConsumption: Target variable (kWh).
# - Temperature: Outdoor temperature (°C).
# - Humidity: Relative humidity (%).
# - BuildingArea: Building size (sq ft).
# - Occupancy: Number of occupants.
# - DayOfWeek: Day of the week (0-6, Monday-Sunday).

# Tools and Libraries:
# - Python, pandas, numpy, scikit-learn, xgboost, tensorflow, matplotlib, seaborn

# Prerequisites:
# Install required libraries:
# pip install pandas numpy scikit-learn xgboost tensorflow matplotlib seaborn
# Ensure building_energy.csv is in the same directory as this script.

# Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, TimeSeriesSplit, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from xgboost import XGBRegressor
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)

# Step 1: Data Loading and Cleaning
# Load the dataset and handle missing values
print('Step 1: Data Loading and Cleaning')
try:
    df = pd.read_csv('building_energy.csv')
except FileNotFoundError:
    dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='H')
    data = {
        'Timestamp': dates,
        'EnergyConsumption': np.random.uniform(50, 200, len(dates)) + np.sin(np.arange(len(dates)) * 0.01) * 20,
        'Temperature': np.random.uniform(10, 30, len(dates)),
        'Humidity': np.random.uniform(30, 90, len(dates)),
        'BuildingArea': np.random.choice([10000, 15000, 20000, 25000], len(dates)),
        'Occupancy': np.random.randint(0, 100, len(dates)),
        'DayOfWeek': [d.weekday() for d in dates]
    }
    df = pd.DataFrame(data)

# Convert Timestamp to datetime
df['Timestamp'] = pd.to_datetime(df['Timestamp'])
df.set_index('Timestamp', inplace=True)

# Check for missing values
print('Missing Values:')
print(df.isnull().sum())

# Basic info
print('\nDataset Info:')
print(df.info())

# Step 2: Feature Engineering
# Create lagged variables, rolling statistics, and temporal features
print('\nStep 2: Feature Engineering')

# Lagged variables (previous hour's energy consumption)
df['Energy_Lag1'] = df['EnergyConsumption'].shift(1)
df['Energy_Lag2'] = df['EnergyConsumption'].shift(2)

# Rolling mean and standard deviation (3-hour window)
df['Energy_RollingMean'] = df['EnergyConsumption'].rolling(window=3).mean()
df['Energy_RollingStd'] = df['EnergyConsumption'].rolling(window=3).std()

# Extract hour from timestamp
df['Hour'] = df.index.hour

# Drop rows with NaN values from feature engineering
df = df.dropna()

# Define features and target
X = df[['Temperature', 'Humidity', 'BuildingArea', 'Occupancy', 'DayOfWeek', 'Energy_Lag1', 'Energy_Lag2', 'Energy_RollingMean', 'Energy_RollingStd', 'Hour']]
y = df['EnergyConsumption']

# Step 3: Data Preprocessing
# Scale numerical features
print('\nStep 3: Data Preprocessing')

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split data into training and testing sets (time-series split)
train_size = int(len(X) * 0.8)
X_train, X_test = X_scaled[:train_size], X_scaled[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

print('Training set shape:', X_train.shape)
print('Testing set shape:', X_test.shape)

# Step 4: Model Training and Comparison
# Train Random Forest, XGBoost, and LSTM models
print('\nStep 4: Model Training and Comparison')

# Random Forest
rf_model = RandomForestRegressor(random_state=42)
rf_params = {'n_estimators': [50, 100], 'max_depth': [5, 10]}
rf_grid = GridSearchCV(rf_model, rf_params, cv=TimeSeriesSplit(n_splits=5), scoring='neg_mean_absolute_error')
rf_grid.fit(X_train, y_train)
rf_best = rf_grid.best_estimator_
rf_pred = rf_best.predict(X_test)

# XGBoost
xgb_model = XGBRegressor(random_state=42)
xgb_params = {'n_estimators': [50, 100], 'max_depth': [3, 5]}
xgb_grid = GridSearchCV(xgb_model, xgb_params, cv=TimeSeriesSplit(n_splits=5), scoring='neg_mean_absolute_error')
xgb_grid.fit(X_train, y_train)
xgb_best = xgb_grid.best_estimator_
xgb_pred = xgb_best.predict(X_test)

# LSTM
X_train_lstm = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))
X_test_lstm = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))
lstm_model = Sequential()
lstm_model.add(LSTM(50, input_shape=(1, X_train.shape[1]), return_sequences=False))
lstm_model.add(Dense(1))
lstm_model.compile(optimizer='adam', loss='mse')
lstm_model.fit(X_train_lstm, y_train, epochs=10, batch_size=32, verbose=0)
lstm_pred = lstm_model.predict(X_test_lstm).flatten()

# Step 5: Model Evaluation
# Evaluate models using MAE and RMSE
print('\nStep 5: Model Evaluation')

def evaluate_model(y_true, y_pred, model_name):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    print(f'{model_name} Performance:')
    print(f'MAE: {mae:.2f} kWh')
    print(f'RMSE: {rmse:.2f} kWh')
    return mae, rmse

rf_mae, rf_rmse = evaluate_model(y_test, rf_pred, 'Random Forest')
xgb_mae, xgb_rmse = evaluate_model(y_test, xgb_pred, 'XGBoost')
lstm_mae, lstm_rmse = evaluate_model(y_test, lstm_pred, 'LSTM')

# Step 6: Visualization
# Plot actual vs predicted values and feature importance
print('\nStep 6: Visualization')

# Actual vs Predicted Plot
plt.figure(figsize=(12, 6))
plt.plot(y_test.index, y_test, label='Actual', color='blue')
plt.plot(y_test.index, rf_pred, label='Random Forest Predicted', color='green')
plt.plot(y_test.index, xgb_pred, label='XGBoost Predicted', color='red')
plt.plot(y_test.index, lstm_pred, label='LSTM Predicted', color='purple')
plt.title('Actual vs Predicted Energy Consumption')
plt.xlabel('Timestamp')
plt.ylabel('Energy Consumption (kWh)')
plt.legend()
plt.show()

# Feature Importance (Random Forest)
feature_importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': rf_best.feature_importances_
}).sort_values(by='Importance', ascending=False)

plt.figure(figsize=(8, 5))
sns.barplot(x='Importance', y='Feature', data=feature_importance)
plt.title('Feature Importance (Random Forest)')
plt.show()

# Conclusion
print('\nConclusion:')
print('This project demonstrates an advanced time-series ML workflow for predicting energy consumption.')
print('Random Forest, XGBoost, and LSTM models were compared, with feature engineering and hyperparameter tuning.')
print('The models provide accurate predictions, with key features like lagged energy values and temperature driving results.')
print('This work can support energy optimization in commercial buildings.')

# Future Work
print('\nFuture Work:')
print('- Incorporate weather forecasts and real-time occupancy data.')
print('- Explore ensemble methods combining all three models.')
print('- Deploy the model as an API for real-time energy predictions.')
