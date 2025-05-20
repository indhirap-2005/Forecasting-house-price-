import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb

st.title("House Price Prediction App")

# Upload CSV file
uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    
    st.write("Dataset Preview:")
    st.dataframe(data.head())

    # Handle missing values
    data.fillna(data.median(numeric_only=True), inplace=True)
    categorical_columns = data.select_dtypes(include=['object']).columns
    data[categorical_columns] = data[categorical_columns].fillna('Missing')

    # Encode categorical columns
    for col in categorical_columns:
        le = LabelEncoder()
        data[col] = le.fit_transform(data[col])

    if 'SalePrice' in data.columns:
        X = data.drop(columns=['SalePrice'])
        y = data['SalePrice']

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Scale data
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        # Train models
        rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
        rf_model.fit(X_train, y_train)

        xgb_model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, random_state=42)
        xgb_model.fit(X_train, y_train)

        # Evaluation function
        def evaluate_model(model, X_test, y_test):
            predictions = model.predict(X_test)
            mae = mean_absolute_error(y_test, predictions)
            mse = mean_squared_error(y_test, predictions)
            rmse = np.sqrt(mse)
            r2 = r2_score(y_test, predictions)
            return mae, mse, rmse, r2

        # Show results
        st.subheader("Model Evaluation")

        rf_results = evaluate_model(rf_model, X_test, y_test)
        xgb_results = evaluate_model(xgb_model, X_test, y_test)

        st.write("**Random Forest Regressor:**")
        st.write(f"MAE: {rf_results[0]:.2f}")
        st.write(f"MSE: {rf_results[1]:.2f}")
        st.write(f"RMSE: {rf_results[2]:.2f}")
        st.write(f"R² Score: {rf_results[3]:.2f}")

        st.write("**XGBoost Regressor:**")
        st.write(f"MAE: {xgb_results[0]:.2f}")
        st.write(f"MSE: {xgb_results[1]:.2f}")
        st.write(f"RMSE: {xgb_results[2]:.2f}")
        st.write(f"R² Score: {xgb_results[3]:.2f}")
    else:
        st.error("The dataset must contain a 'SalePrice' column.")
