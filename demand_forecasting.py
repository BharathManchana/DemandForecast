import pandas as pd
import numpy as np
from prophet import Prophet
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller
import statsmodels.api as sm

# Load data with caching to optimize loading time
@st.cache_data
def load_data():
    try:
        transactions_df = pd.read_csv(r'C:\Users\Bharath\Desktop\demand_forecasting\Transactional_data_retail_02.csv')
        product_info_df = pd.read_csv(r'C:\Users\Bharath\Desktop\demand_forecasting\ProductInfo.csv')
        customer_df = pd.read_csv(r'C:\Users\Bharath\Desktop\demand_forecasting\CustomerDemographics.csv')

        return transactions_df, product_info_df, customer_df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None, None, None

# Preprocess and filter data based on transaction dates
def preprocess_data(transactions_df):
    transactions_df['InvoiceDate'] = pd.to_datetime(transactions_df['InvoiceDate'], format='%d-%m-%Y')
    transactions_df['Customer ID'] = transactions_df['Customer ID'].fillna(0)
    start_date = '2022-01-01'
    end_date = '2023-12-31'
    transactions_df = transactions_df[(transactions_df['InvoiceDate'] >= start_date) & 
                                       (transactions_df['InvoiceDate'] <= end_date)]
    transactions_df['Week'] = transactions_df['InvoiceDate'].dt.to_period('W').apply(lambda r: r.start_time)
    return transactions_df

# Exploratory Data Analysis (EDA)
def perform_eda(transactions_df, product_info_df, customer_df):
    st.subheader("Exploratory Data Analysis (EDA)")

    # Customer-level summary statistics
    customer_summary = customer_df.describe()
    st.write("### Customer Summary Statistics")
    st.dataframe(customer_summary)

    # Item-level summary statistics
    product_summary = product_info_df.describe()
    st.write("### Product Summary Statistics")
    st.dataframe(product_summary)

    # Transaction-level summary statistics
    transaction_summary = transactions_df.describe()
    st.write("### Transaction Summary Statistics")
    st.dataframe(transaction_summary)

    # Visualizations
    st.subheader("Visualizations")

    # Revenue distribution
    transactions_df['Revenue'] = transactions_df['Quantity'] * transactions_df['Price']
    plt.figure(figsize=(10, 5))
    sns.histplot(transactions_df['Revenue'], bins=30, kde=True)
    st.pyplot(plt.gcf())  # Clear the figure after plotting

# Calculate top 10 products by revenue and quantity
def calculate_top_10_products(transactions_df):
    transactions_df['Revenue'] = transactions_df['Quantity'] * transactions_df['Price']
    top_10_quantity = transactions_df.groupby('StockCode')['Quantity'].sum().sort_values(ascending=False).head(10)
    top_10_revenue = transactions_df.groupby('StockCode')['Revenue'].sum().sort_values(ascending=False).head(10)
    return top_10_quantity, top_10_revenue

# SQL-like join queries (simulated with pandas)
def join_queries(transactions_df, product_info_df, customer_df):
    # Simulate SQL joins
    transactions_with_product = pd.merge(transactions_df, product_info_df, on='StockCode', how='left')
    transactions_with_customer = pd.merge(transactions_with_product, customer_df, on='Customer ID', how='left')
    return transactions_with_customer

# Prepare data for XGBoost forecasting
def prepare_data_for_xgboost(stock_code_data):
    stock_code_data = stock_code_data.copy()
    stock_code_data['Month'] = pd.to_datetime(stock_code_data['Week'])
    stock_code_data['Month_num'] = stock_code_data['Month'].dt.month
    stock_code_data['Year'] = stock_code_data['Month'].dt.year

    stock_code_data['lag_1'] = stock_code_data['Quantity'].shift(1)
    stock_code_data['lag_2'] = stock_code_data['Quantity'].shift(2)
    stock_code_data['lag_4'] = stock_code_data['Quantity'].shift(4)

    numeric_columns = stock_code_data.select_dtypes(include=[np.number]).columns
    stock_code_data[numeric_columns] = stock_code_data[numeric_columns].fillna(stock_code_data[numeric_columns].mean())
    
    return stock_code_data[['Month_num', 'Year', 'lag_1', 'lag_2', 'lag_4', 'Quantity']]

# Prophet Forecasting Function with Better Error Handling
def forecast_with_prophet(stock_code_data, stock_code, periods=15):
    df_prophet = stock_code_data.rename(columns={'Week': 'ds', 'Quantity': 'y'})
    df_prophet['ds'] = pd.to_datetime(df_prophet['ds'])
    df_prophet = df_prophet[df_prophet['y'] > 0]

    # Check if there are enough data points for Prophet
    if df_prophet.shape[0] < 10:
        st.error(f"Not enough data for Prophet forecasting for stock code: {stock_code}")
        return pd.DataFrame()

    # Filter out extreme outliers
    threshold = df_prophet['y'].quantile(0.90)  # Adjust threshold if necessary
    df_prophet = df_prophet[df_prophet['y'] <= threshold]

    if df_prophet.empty or df_prophet['y'].count() < 5:
        st.error(f"No data after filtering for Prophet forecasting for stock code: {stock_code}")
        return pd.DataFrame()

    df_prophet['y'] = np.log1p(df_prophet['y'])  # Log transformation

    model = Prophet(yearly_seasonality=False, weekly_seasonality=True, daily_seasonality=False)

    try:
        model.fit(df_prophet)
        future = model.make_future_dataframe(periods=periods, freq='W')
        forecast = model.predict(future)
        forecast['yhat'] = np.expm1(forecast['yhat'])  # Inverse log transformation
        return forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]
    except Exception as e:
        st.error(f"Prophet model fitting failed for stock code {stock_code}. Error: {e}")
        return pd.DataFrame()

# XGBoost Forecasting with Better Error Handling
def forecast_with_xgboost(stock_code_data, forecast_horizon=15):
    stock_data = prepare_data_for_xgboost(stock_code_data)
    X = stock_data[['Month_num', 'Year', 'lag_1', 'lag_2', 'lag_4']]
    y = stock_data['Quantity']

    # Debugging step: Check if stock code data is available
    st.write(f"Stock code data size:", stock_code_data.shape)
    st.write(stock_code_data.head())

    if len(X) > 1:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        xgb_model = XGBRegressor(objective='reg:squarederror', n_estimators=50, random_state=42)
        xgb_model.fit(X_train, y_train)

        max_month = stock_data['Month_num'].max()
        future_months = pd.DataFrame({
            'Month_num': range(max_month + 1, max_month + 1 + forecast_horizon),
            'Year': [stock_data['Year'].max()] * forecast_horizon,
            'lag_1': [y_train.iloc[-1]] * forecast_horizon,
            'lag_2': [y_train.iloc[-2]] * forecast_horizon,
            'lag_4': [y_train.iloc[-4]] * forecast_horizon
        })

        predictions = xgb_model.predict(future_months)

        y_pred = xgb_model.predict(X_test)
        errors = y_test - y_pred

        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))

        return predictions, mae, rmse
    else:
        st.error("Not enough data for XGBoost forecasting.")
        return None, None, None

# Plot ACF and PACF
def plot_acf_pacf(stock_code_data):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    
    # ACF Plot
    plot_acf(stock_code_data['Quantity'], ax=ax1, lags=20)
    ax1.set_title('Auto-Correlation Function (ACF)')
    ax1.set_xlabel('Lags')
    ax1.set_ylabel('ACF Values')
    
    # PACF Plot
    plot_pacf(stock_code_data['Quantity'], ax=ax2, lags=20)
    ax2.set_title('Partial Auto-Correlation Function (PACF)')
    ax2.set_xlabel('Lags')
    ax2.set_ylabel('PACF Values')

    st.pyplot(fig)

# Time Series Analysis Function for ARIMA/ETS Models
def time_series_analysis(stock_code_data, stock_code):
    # Fit ARIMA model
    from statsmodels.tsa.arima.model import ARIMA
    import warnings
    warnings.filterwarnings("ignore")

    model = ARIMA(stock_code_data['Quantity'], order=(1, 1, 1))
    fitted_model = model.fit()
    st.write(f"ARIMA model summary for stock code {stock_code}:")
    st.write(fitted_model.summary())

# Main app layout and execution
def main():
    st.title("Demand Forecasting App")

    # Load and preprocess data
    transactions_df, product_info_df, customer_df = load_data()
    if transactions_df is None:
        return

    transactions_df = preprocess_data(transactions_df)
    perform_eda(transactions_df, product_info_df, customer_df)

    # Calculate top products
    top_10_quantity, top_10_revenue = calculate_top_10_products(transactions_df)

    # User inputs
    stock_code = st.selectbox("Select Stock Code", top_10_quantity.index)
    forecast_horizon = st.number_input("Forecast Horizon (weeks)", min_value=1, max_value=52, value=15)

    # Get data for the selected stock code
    stock_code_data = transactions_df[transactions_df['StockCode'] == stock_code].groupby('Week').agg({'Quantity': 'sum'}).reset_index()

    # ACF and PACF plots
    plot_acf_pacf(stock_code_data)

    # Forecasting
    if st.button("Forecast with Prophet"):
        prophet_forecast = forecast_with_prophet(stock_code_data, stock_code, forecast_horizon)
        if not prophet_forecast.empty:
            st.write("### Prophet Forecast")
            st.line_chart(prophet_forecast.set_index('ds')['yhat'])

    if st.button("Forecast with XGBoost"):
        predictions, mae, rmse = forecast_with_xgboost(stock_code_data, forecast_horizon)
        if predictions is not None:
            st.write("### XGBoost Forecast")
            st.line_chart(predictions)

            st.write(f"Mean Absolute Error: {mae}")
            st.write(f"Root Mean Squared Error: {rmse}")

    if st.button("Time Series Analysis (ARIMA)"):
        time_series_analysis(stock_code_data, stock_code)

if __name__ == "__main__":
    main()
