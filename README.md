# EnergyPredict
Energy Consumption Prediction

This repository contains a Machine Learning project to predict hourly energy consumption in commercial buildings using time-series data and models like Random Forest, XGBoost, and LSTM. The project demonstrates an advanced ML workflow, including feature engineering, hyperparameter tuning, cross-validation, and visualization.

Project Overview





Objective: Predict hourly energy consumption using features like temperature, humidity, building area, and occupancy.



Dataset: Synthetic time-series dataset (building_energy.csv) with hourly observations.



Models: Random Forest, XGBoost, and LSTM, achieving low MAE and RMSE.



Tools: Python, pandas, numpy, scikit-learn, xgboost, tensorflow, matplotlib, seaborn.


Installation





Clone the repository:

git clone https://github.com/yourusername/EnergyPredict.git



Install dependencies:

pip install pandas numpy scikit-learn xgboost tensorflow matplotlib seaborn



Results





Achieved low MAE and RMSE across models, with Random Forest and XGBoost performing best.



Identified key features (lagged energy values, temperature) driving predictions.



Visualized actual vs predicted values and feature importance for actionable insights.

Future Work





Incorporate weather forecasts and real-time occupancy data.



Explore ensemble methods combining all three models.



Deploy the model as an API for real-time predictions.
