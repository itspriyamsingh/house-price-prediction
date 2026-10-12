# House Price Prediction — Streamlit Community Cloud

A deploy-ready Streamlit machine-learning app based on the commonly distributed **Housing Prices** dataset (545 rows, 13 columns including the target). It uses the original feature set:

- Numeric: `area`, `bedrooms`, `bathrooms`, `stories`, `parking`
- Yes/no: `mainroad`, `guestroom`, `basement`, `hotwaterheating`, `airconditioning`, `prefarea`
- Category: `furnishingstatus`
- Target: `price`

The app preprocesses categorical features with scikit-learn pipelines, trains Linear Regression and Random Forest models, compares MAE/RMSE/R² on a fixed 80/20 split, and lets users make predictions.

## Files

```text
house-price-prediction-streamlit/
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
└── packages.txt
```

## Deploy to Streamlit Community Cloud

1. Download and extract this package.
2. Create a new **public or private GitHub repository** and upload the files above.
3. Open https://share.streamlit.io/ and sign in with GitHub.
4. Select **Create app**.
5. Select your repository, branch (usually `main`), and main file path `app.py`.
6. Click **Deploy**. Streamlit installs dependencies from `requirements.txt`.
7. When deployment completes, open the generated app URL.

The app loads `Housing.csv` from the project directory if you add it. If the file is not present, it fetches the public CSV from the source URL in `app.py`. For more reliable deployments, you can download the dataset from the Kaggle source below and commit `Housing.csv` alongside `app.py`.

## Run locally (optional)

```bash
py -m venv .venv
.venv\Scripts\activate
py -m pip install -r requirements.txt
py -m streamlit run app.py
```

On macOS/Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

## Dataset source

- Kaggle: https://www.kaggle.com/datasets/yasserh/housing-prices-dataset
- Public CSV mirror used as fallback: https://raw.githubusercontent.com/shahndaa/Housing-prices/main/Housing.csv

The app validates the required columns. If the remote source is unavailable, add `Housing.csv` to this folder and redeploy.

## Model and data notes

- Models are trained in the app from the dataset at startup and cached for the session; there is no external `.pkl` file required.
- The model is retrained when Streamlit's cache is cleared or the app code/data changes.
- This is an educational dataset, not a complete representation of current real estate markets. Predictions are estimates, not professional valuations.
- The app reports hold-out metrics but does not claim the models are production validated.
