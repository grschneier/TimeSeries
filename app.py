import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, accuracy_score, classification_report

# ------------------------------
# Load Data
# ------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv('tsla_data.csv', index_col=0, parse_dates=True)

    # Rename columns for clarity
    df.rename(columns={'1. open': 'open', '2. high': 'high', '3. low': 'low',
                       '4. close': 'close', '5. volume': 'volume'}, inplace=True)

    # Ensure chronological order
    df = df.sort_index(ascending=True)
    return df

df = load_data()

# ------------------------------
# Feature Engineering
# ------------------------------
df['open-close'] = df['open'] - df['close']
df['low-high'] = df['low'] - df['high']
df['is_quarter_end'] = df.index.to_series().dt.is_quarter_end.astype(int)

# Create a binary classification target (1 if price goes up, 0 if it goes down)
df['target'] = (df['close'].shift(-1) > df['close']).astype(int)

# Drop NaN values created by shifting
df.dropna(inplace=True)

# ------------------------------
# Sidebar Inputs
# ------------------------------
st.sidebar.header("User Input Settings")
lags = st.sidebar.slider("Number of Lag Features", min_value=1, max_value=10, value=3)
test_size = st.sidebar.slider("Test Data Size (as %)", min_value=5, max_value=50, value=20)

# ------------------------------
# Creating Lag Features
# ------------------------------
for i in range(1, lags + 1):
    df[f'lag_{i}'] = df['close'].shift(i)

# Drop NaNs caused by shifting
df.dropna(inplace=True)

# ------------------------------
# Split Data into Train/Test
# ------------------------------
features = df[['open-close', 'low-high', 'is_quarter_end'] + [f'lag_{i}' for i in range(1, lags + 1)]]
target = df['target']

scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

X_train, X_test, y_train, y_test = train_test_split(
    features_scaled, target, test_size=test_size/100, random_state=2022, shuffle=False)

# ------------------------------
# Train XGBoost Classifier
# ------------------------------
clf = xgb.XGBClassifier(random_state=2024)
clf.fit(X_train, y_train)

# Predictions
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

# ------------------------------
# Feature Importance
# ------------------------------
importance_df = pd.DataFrame({'Feature': features.columns, 'Importance': clf.feature_importances_})
importance_df = importance_df.sort_values(by='Importance', ascending=False)

# ------------------------------
# Streamlit Dashboard
# ------------------------------
st.title("📈 TSLA Stock Price Forecasting with XGBoost")

# Display Data
st.subheader("📊 Stock Data Overview")
st.dataframe(df.head())

# ------------------------------
# Plot 1: Actual vs. Predicted Stock Direction
# ------------------------------
st.subheader("🔮 Actual vs. Predicted Stock Direction")

df_resampled = df.resample('M').mean()
y_test_resampled = df_resampled['target']
y_pred_resampled = pd.Series(y_pred, index=df.index[-len(y_test):]).resample('M').mean()

fig = go.Figure()
fig.add_trace(go.Scatter(x=y_test_resampled.index, y=y_test_resampled, mode='lines+markers', name='Actual', marker=dict(color="blue")))
fig.add_trace(go.Scatter(x=y_pred_resampled.index, y=y_pred_resampled, mode='lines+markers', name='Predicted', marker=dict(color="red")))
fig.update_layout(title="Actual vs. Predicted (Monthly Aggregation)", xaxis_title="Date", yaxis_title="Direction")
st.plotly_chart(fig)

# ------------------------------
# Plot 2: Feature Importance
# ------------------------------
st.subheader("🔥 Feature Importance from XGBoost")
fig_importance = px.bar(importance_df, x='Importance', y='Feature', orientation='h', title="Feature Importance")
st.plotly_chart(fig_importance)

# ------------------------------
# Plot 3: Error Distribution
# ------------------------------
st.subheader("📉 Prediction Error Distribution")
errors = y_test - y_pred
fig_error = px.histogram(errors, nbins=20, title="Distribution of Prediction Errors", labels={'value': 'Error'})
st.plotly_chart(fig_error)

# ------------------------------
# Model Performance Metrics
# ------------------------------
st.subheader("🎯 Model Performance Metrics")
st.metric(label="Test Accuracy", value=f"{accuracy:.2%}")
st.metric(label="Test RMSE", value=f"{rmse:.2f}")

# ------------------------------
# Rolling Mean & Volatility
# ------------------------------
st.subheader("📈 Rolling Averages & Volatility")
df['rolling_mean'] = df['close'].rolling(window=20).mean()
df['volatility'] = df['close'].rolling(window=20).std()

fig_roll = go.Figure()
fig_roll.add_trace(go.Scatter(x=df.index, y=df['close'], mode='lines', name='Closing Price'))
fig_roll.add_trace(go.Scatter(x=df.index, y=df['rolling_mean'], mode='lines', name='20-day Rolling Mean'))
fig_roll.update_layout(title="Stock Price with Rolling Averages", xaxis_title="Date", yaxis_title="Price")
st.plotly_chart(fig_roll)

# ------------------------------
# Download Results
# ------------------------------
st.subheader("📥 Download Predictions")
df_results = pd.DataFrame({'Date': df.index[-len(y_test):], 'Actual': y_test, 'Predicted': y_pred})
csv = df_results.to_csv(index=False).encode('utf-8')
st.download_button("Download Predictions CSV", data=csv, file_name="xgboost_predictions.csv", mime="text/csv")

# ------------------------------
# Conclusion
# ------------------------------
st.markdown("### ✅ **Key Takeaways**")
st.markdown("""
- The **XGBoost model** predicts stock movement direction (up/down) based on lag features and other indicators.
- The **feature importance plot** shows which factors most influence predictions.
- **Rolling averages & volatility** help identify trends and potential market shifts.
- The **error distribution** gives insight into model accuracy and potential biases.
""")

st.markdown("🚀 Built with Streamlit & Plotly
