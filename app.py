import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import matplotlib.font_manager as fm
from sklearn.pipeline import make_pipeline

# Load the custom font
font_path = '/mnt/data/file-ngwyeoEN29l1M3O1QpdxCwkj'
font_prop = fm.FontProperties(fname=font_path)

# Load the dataset
@st.cache
def load_data():
    data = pd.read_csv('/mnt/data/file-DM5RHVe8asgbMmGW5xPZWi')  # Update with your correct dataset path
    return data

# Display basic statistics and correlations
def analyze_data(data):
    desc = data.describe()
    corr = data.corr()
    return desc, corr

# Visualize price distribution
def plot_price_distribution(data):
    plt.figure(figsize=(10, 5))
    plt.hist(data['price'], bins=30, color='skyblue', edgecolor='black')
    plt.title('House Price Distribution', fontproperties=font_prop)
    plt.xlabel('Price', fontproperties=font_prop)
    plt.ylabel('Frequency', fontproperties=font_prop)
    st.pyplot(plt)

# Visualize price vs square footage
def plot_price_vs_sqft(data):
    plt.figure(figsize=(10, 5))
    plt.scatter(data['sqft_living'], data['price'], alpha=0.5, color='green')
    plt.title('Price vs Living Area', fontproperties=font_prop)
    plt.xlabel('Square Footage (Living Area)', fontproperties=font_prop)
    plt.ylabel('Price', fontproperties=font_prop)
    st.pyplot(plt)

# Feature Engineering: Convert 'yr_built' to 'house_age'
def feature_engineering(data):
    data['house_age'] = 2025 - data['yr_built']
    data = data.drop(columns=['yr_built', 'zipcode', 'lat', 'long'])
    return data

# Predict price based on user inputs
def predict_price(features, model_type='linear'):
    X = features.drop('price', axis=1)
    y = features['price']

    # Scaling the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Splitting the data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    if model_type == 'linear':
        model = LinearRegression()
    elif model_type == 'ridge':
        model = Ridge(alpha=1.0)
    elif model_type == 'lasso':
        model = Lasso(alpha=0.1)
    elif model_type == 'random_forest':
        model = RandomForestRegressor(n_estimators=100, random_state=42)

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return model, y_pred, y_test

# Model Evaluation: Cross-Validation and Error Metrics
def evaluate_model(model, X, y):
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(model, X, y, cv=kf, scoring='neg_mean_squared_error')
    mse = -cv_scores.mean()
    rmse = np.sqrt(mse)
    r2 = cross_val_score(model, X, y, cv=kf, scoring='r2').mean()
    return mse, rmse, r2

# 3D Visualization for Price vs Features
def plot_3d_price_vs_features(data):
    fig = go.Figure(data=[go.Scatter3d(
        x=data['sqft_living'],
        y=data['bedrooms'],
        z=data['price'],
        mode='markers',
        marker=dict(
            size=5,
            color=data['price'],  # Color by price
            colorscale='Viridis',
            opacity=0.8
        ),
        text=data.apply(lambda row: f'Price: ${row["price"]:,.0f}<br>Sqft: {row["sqft_living"]}<br>Bedrooms: {row["bedrooms"]}', axis=1)
    )])

    fig.update_layout(
        scene=dict(
            xaxis_title='Living Area (Sqft)',
            yaxis_title='Number of Bedrooms',
            zaxis_title='Price',
        ),
        title="3D View: Price vs Living Area vs Bedrooms",
        coloraxis_colorbar=dict(
            title="Price"
        )
    )

    st.plotly_chart(fig)

# Loan Payment Calculator
def calculate_monthly_payment(loan_amount, interest_rate, years):
    monthly_rate = interest_rate / 100 / 12
    number_of_payments = years * 12
    monthly_payment = loan_amount * (monthly_rate * (1 + monthly_rate) ** number_of_payments) / \
                      ((1 + monthly_rate) ** number_of_payments - 1)
    return monthly_payment

# Main function
def main():
    st.title('Advanced House Price Prediction Tool by Muhammad Allam Rafi')

    # Load data
    data = load_data()

    # Display dataset overview
    st.subheader('Dataset Overview')
    st.write(data.head())

    # Perform feature engineering
    data = feature_engineering(data)

    # Display summary statistics
    st.subheader('Summary Statistics')
    desc, corr = analyze_data(data)
    st.write(desc)

    # Show correlation heatmap
    st.subheader('Correlation Matrix')
    plt.figure(figsize=(10, 6))
    sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', linewidths=0.5)
    plt.title('Correlation Matrix', fontsize=16, fontproperties=font_prop)
    st.pyplot(plt)

    # Visualizations
    st.subheader('Visualizations')
    plot_price_distribution(data)
    plot_price_vs_sqft(data)

    # 3D Price Prediction Visualization
    st.subheader('3D Price Prediction Visualization')
    plot_3d_price_vs_features(data)

    # Property Price Prediction
    st.subheader('Property Price Prediction')
    sqft_living = st.number_input('Enter square footage of living area:', min_value=100, max_value=10000)
    bedrooms = st.number_input('Enter number of bedrooms:', min_value=1, max_value=10)
    bathrooms = st.number_input('Enter number of bathrooms:', min_value=1, max_value=10)
    floors = st.number_input('Enter number of floors:', min_value=1, max_value=3)

    custom_features = pd.DataFrame({
        'sqft_living': [sqft_living],
        'bedrooms': [bedrooms],
        'bathrooms': [bathrooms],
        'floors': [floors]
    })

    model_type = st.selectbox('Choose model type for prediction:', ['linear', 'ridge', 'lasso', 'random_forest'])

    if st.button('Predict Price'):
        model, y_pred, y_test = predict_price(data, model_type)
        mse, rmse, r2 = evaluate_model(model, data.drop('price', axis=1), data['price'])
        predicted_price = model.predict(custom_features)[0]
        st.write(f'Estimated Price: ${predicted_price:.2f}')
        st.write(f'Mean Squared Error: {mse:.2f}')
        st.write(f'Root Mean Squared Error: {rmse:.2f}')
        st.write(f'R-squared: {r2:.2f}')

    # Loan Payment Calculator
    st.subheader('KPR (Cicilan) Calculator')
    loan_amount = st.number_input('Enter loan amount:', min_value=0)
    interest_rate = st.number_input('Enter interest rate (%):', min_value=0.0, max_value=100.0)
    years = st.number_input('Enter loan term (years):', min_value=1, max_value=30)

    if st.button('Calculate Monthly Payment'):
        if loan_amount > 0 and interest_rate > 0 and years > 0:
            monthly_payment = calculate_monthly_payment(loan_amount, interest_rate, years)
            st.write(f'Your monthly payment will be: ${monthly_payment:.2f}')
        else:
            st.write('Please enter valid inputs for loan amount, interest rate, and years.')

    # Watermark
    st.write('---')
    st.write('Created by Muhammad Allam Rafi')

if __name__ == "__main__":
    main()
