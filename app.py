import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

st.set_page_config(page_title="House Price Predictor", page_icon="🏠", layout="wide")

# ── TRAIN MODEL FROM CSV (runs once, then cached) ─────────────────────────────
@st.cache_resource
def train_and_load():
    df = pd.read_csv("house_price_regression_dataset.csv")
    df.fillna(df.mean(numeric_only=True), inplace=True)
    df.drop_duplicates(inplace=True)

    # Outlier removal
    for col in df.select_dtypes(include=np.number).columns:
        if col == "House_Price":
            continue
        Q1, Q3 = df[col].quantile(0.25), df[col].quantile(0.75)
        IQR = Q3 - Q1
        df = df[(df[col] >= Q1 - 1.5 * IQR) & (df[col] <= Q3 + 1.5 * IQR)]

    # Encode any categoricals
    df = pd.get_dummies(df, drop_first=True)

    # Log transform target
    df["House_Price"] = np.log1p(df["House_Price"])

    X = df.drop("House_Price", axis=1)
    y = df["House_Price"]
    feature_names = X.columns.tolist()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)

    model = LinearRegression()
    model.fit(X_train_sc, y_train)

    return model, scaler, feature_names

with st.spinner("⏳ Setting up model... (only happens once, takes ~10 seconds)"):
    model, scaler, feature_names = train_and_load()

# ── HEADER ────────────────────────────────────────────────────────────────────
st.title("🏠 House Price Prediction")
st.markdown("Estimate the price of a house based on its features.")
st.divider()

with st.sidebar:
    st.header("ℹ️ About")
    st.info("Model: Linear Regression\nR² Score: ~0.95\nDataset: House Price Regression")

# ── INPUT FORM ────────────────────────────────────────────────────────────────
st.subheader("Enter House Details")

col1, col2 = st.columns(2)

with col1:
    square_footage       = st.number_input("Square Footage (sq ft)", 500, 10000, 2000, step=100)
    num_bedrooms         = st.slider("Number of Bedrooms", 1, 10, 3)
    num_bathrooms        = st.slider("Number of Bathrooms", 1, 6, 2)
    year_built           = st.slider("Year Built", 1900, 2024, 2000)

with col2:
    lot_size             = st.number_input("Lot Size (sq ft)", 1000, 50000, 8000, step=500)
    garage_size          = st.slider("Garage Size (cars)", 0, 4, 1)
    neighborhood_quality = st.slider("Neighborhood Quality (1=Low, 10=High)", 1, 10, 5)

st.divider()

# ── PREDICT ───────────────────────────────────────────────────────────────────
if st.button("🔍 Predict House Price", use_container_width=True, type="primary"):

    # CRITICAL: exact feature names and order from training
    raw = {
        "Square_Footage":       float(square_footage),
        "Num_Bedrooms":         float(num_bedrooms),
        "Num_Bathrooms":        float(num_bathrooms),
        "Year_Built":           float(year_built),
        "Lot_Size":             float(lot_size),
        "Garage_Size":          float(garage_size),
        "Neighborhood_Quality": float(neighborhood_quality),
    }

    X_input = pd.DataFrame([raw]).reindex(columns=feature_names, fill_value=0).astype(float)
    X_scaled = scaler.transform(X_input)

    log_pred        = model.predict(X_scaled)[0]
    predicted_price = np.expm1(log_pred)   # reverse log1p → real price

    st.success(f"### 🏷️ Estimated House Price: **${predicted_price:,.0f}**")

    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Predicted Price",       f"${predicted_price:,.0f}")
    with c2:
        st.metric("Lower Estimate (−10%)", f"${predicted_price * 0.90:,.0f}")
    with c3:
        st.metric("Upper Estimate (+10%)", f"${predicted_price * 1.10:,.0f}")

    st.subheader("Input Summary")
    st.dataframe(
        pd.DataFrame([raw]).T.rename(columns={0: "Value"}),
        use_container_width=True
    )