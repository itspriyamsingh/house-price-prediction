from pathlib import Path
from urllib.error import URLError
from urllib.request import urlopen

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

st.set_page_config(page_title="House Price Prediction", page_icon="🏡", layout="wide")

st.title("🏡 House Price Prediction")
st.write(
    "Estimate a house price from its size, layout, location-related features, "
    "and furnishing status. This app trains a model from the public Housing dataset."
)

DATA_PATH = Path(__file__).parent / "Housing.csv"
DATA_URL = (
    "https://raw.githubusercontent.com/shahndaa/Housing-prices/main/Housing.csv"
)
TARGET = "price"
NUMERIC_FEATURES = ["area", "bedrooms", "bathrooms", "stories", "parking"]
BINARY_FEATURES = [
    "mainroad", "guestroom", "basement", "hotwaterheating",
    "airconditioning", "prefarea"
]
CATEGORICAL_FEATURES = ["furnishingstatus"]
FEATURES = NUMERIC_FEATURES + BINARY_FEATURES + CATEGORICAL_FEATURES
EXPECTED_COLUMNS = [TARGET] + FEATURES


@st.cache_data(show_spinner="Loading the housing dataset...")
def load_data():
    if DATA_PATH.exists():
        df = pd.read_csv(DATA_PATH)
    else:
        # The dataset is also available from the source URL below.
        with urlopen(DATA_URL, timeout=20) as response:
            df = pd.read_csv(response)
    missing = sorted(set(EXPECTED_COLUMNS) - set(df.columns))
    if missing:
        raise ValueError(f"Dataset is missing required columns: {missing}")
    df = df[EXPECTED_COLUMNS].dropna().copy()
    for col in BINARY_FEATURES:
        df[col] = df[col].astype(str).str.strip().str.lower()
        df[col] = df[col].map({"yes": "yes", "no": "no"})
    df["furnishingstatus"] = df["furnishingstatus"].astype(str).str.strip().str.lower()
    return df


@st.cache_resource(show_spinner="Training the prediction models...")
def train_models(data):
    X = data[FEATURES]
    y = data[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    numeric_transformer = "passthrough"
    categorical_transformer = OneHotEncoder(handle_unknown="ignore")
    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", numeric_transformer, NUMERIC_FEATURES),
            ("binary", OneHotEncoder(handle_unknown="ignore"), BINARY_FEATURES),
            ("categorical", categorical_transformer, CATEGORICAL_FEATURES),
        ],
        remainder="drop",
    )

    candidates = {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(
            n_estimators=300, random_state=42, n_jobs=-1, min_samples_leaf=2
        ),
    }
    fitted = {}
    rows = []
    for name, estimator in candidates.items():
        pipeline = Pipeline([("preprocessor", preprocessor), ("model", estimator)])
        pipeline.fit(X_train, y_train)
        pred = pipeline.predict(X_test)
        fitted[name] = pipeline
        rows.append({
            "Model": name,
            "MAE": mean_absolute_error(y_test, pred),
            "RMSE": float(np.sqrt(mean_squared_error(y_test, pred))),
            "R²": r2_score(y_test, pred),
        })
    metrics = pd.DataFrame(rows).sort_values("RMSE").reset_index(drop=True)
    best_name = str(metrics.iloc[0]["Model"])
    return fitted, metrics, best_name, len(X_train), len(X_test)


try:
    data = load_data()
except (URLError, OSError, ValueError, pd.errors.ParserError) as exc:
    st.error("The dataset could not be loaded.")
    st.write("Check your internet connection or place `Housing.csv` next to `app.py`.")
    st.code(f"Dataset source: {DATA_URL}")
    st.exception(exc)
    st.stop()

try:
    models, metrics, best_name, train_count, test_count = train_models(data)
except Exception as exc:
    st.error("The model training step failed.")
    st.exception(exc)
    st.stop()

with st.sidebar:
    st.header("Dataset overview")
    st.metric("Rows available", f"{len(data):,}")
    st.metric("Input features", len(FEATURES))
    st.caption("Dataset: Housing Prices (545 records in the commonly distributed version).")
    st.caption("Source: Yasserh Housing Prices Dataset; linked to the public CSV in the README.")

tab_predict, tab_metrics, tab_data = st.tabs(
    ["🔮 Predict price", "📊 Model evaluation", "🗂️ Dataset preview"]
)

with tab_predict:
    st.subheader("Enter property details")
    with st.form("prediction_form"):
        col1, col2, col3 = st.columns(3)
        with col1:
            area = st.number_input("Area (square feet)", min_value=100, max_value=50000, value=5000, step=100)
            bedrooms = st.number_input("Bedrooms", min_value=1, max_value=10, value=3, step=1)
            bathrooms = st.number_input("Bathrooms", min_value=1, max_value=10, value=2, step=1)
            stories = st.number_input("Stories", min_value=1, max_value=10, value=2, step=1)
        with col2:
            parking = st.number_input("Parking spaces", min_value=0, max_value=10, value=1, step=1)
            mainroad = st.selectbox("Main road access", ["yes", "no"])
            guestroom = st.selectbox("Guest room", ["no", "yes"])
            basement = st.selectbox("Basement", ["no", "yes"])
        with col3:
            hotwaterheating = st.selectbox("Hot-water heating", ["no", "yes"])
            airconditioning = st.selectbox("Air conditioning", ["yes", "no"])
            prefarea = st.selectbox("Preferred area", ["yes", "no"])
            furnishingstatus = st.selectbox(
                "Furnishing status", ["furnished", "semi-furnished", "unfurnished"]
            )
        selected_model = st.selectbox("Prediction model", list(models.keys()), index=list(models.keys()).index(best_name))
        submitted = st.form_submit_button("Predict house price", type="primary")

    if submitted:
        row = pd.DataFrame([{
            "area": area,
            "bedrooms": bedrooms,
            "bathrooms": bathrooms,
            "stories": stories,
            "parking": parking,
            "mainroad": mainroad,
            "guestroom": guestroom,
            "basement": basement,
            "hotwaterheating": hotwaterheating,
            "airconditioning": airconditioning,
            "prefarea": prefarea,
            "furnishingstatus": furnishingstatus,
        }], columns=FEATURES)
        prediction = float(models[selected_model].predict(row)[0])
        st.success(f"Estimated price: **{prediction:,.0f}**")
        st.caption("The estimate is based on the training dataset and is not a formal valuation or guarantee.")

with tab_metrics:
    st.subheader("Hold-out test performance")
    st.caption("Models are trained with an 80/20 train-test split and random_state=42.")
    st.dataframe(
        metrics.style.format({"MAE": "{:,.0f}", "RMSE": "{:,.0f}", "R²": "{:.3f}"}),
        use_container_width=True,
        hide_index=True,
    )
    st.write(f"Best model by RMSE on this run: **{best_name}**")
    st.write(f"Training rows: {train_count:,} · Test rows: {test_count:,}")
    st.caption("Metrics describe this split only; they do not guarantee performance on other markets or future properties.")

with tab_data:
    st.subheader("Dataset preview")
    st.dataframe(data.head(20), use_container_width=True, hide_index=True)
    st.download_button(
        "Download loaded dataset as CSV",
        data=data.to_csv(index=False).encode("utf-8"),
        file_name="Housing.csv",
        mime="text/csv",
    )

st.divider()
st.caption(
    "Educational machine-learning project. Dataset source: "
    "https://www.kaggle.com/datasets/yasserh/housing-prices-dataset"
)
