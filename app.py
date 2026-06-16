
import streamlit as st
import pandas as pd
import numpy as np
import altair as alt

from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


# ============================================================
# STREAMLIT PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Argentina Inflation Forecasting",
    page_icon="📈",
    layout="wide"
)

st.title("📈 Forecasting Monthly Inflation in Argentina")
st.subheader("BCRA REM vs. Python Models")

st.markdown(
    """
This mini project estimates Argentina's **next-month monthly inflation** and compares
Python-based forecasting models against the **BCRA REM expectation**.

The app is designed for portfolio use: it can run immediately with simulated demo data,
or you can upload your own CSV with official/processed data from INDEC, BCRA and REM.
"""
)


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def make_demo_data(seed: int = 42) -> pd.DataFrame:
    """
    Creates simulated monthly macroeconomic data.
    This is NOT real Argentina data. It is only for testing the app.
    """
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2017-01-01", "2026-05-01", freq="MS")
    n = len(dates)

    inflation = []
    fx_rate = []
    policy_rate = []
    monetary_base = []
    reserves = []
    rem_expectation = []

    infl = 1.6
    fx = 16.0
    rate = 27.0
    mb = 1000.0
    res = 45000.0

    for i in range(n):
        # Simulated regime changes
        shock = rng.normal(0, 0.45)
        if dates[i].year in [2018, 2019]:
            shock += rng.normal(0.4, 0.5)
        if dates[i].year in [2022, 2023]:
            shock += rng.normal(0.8, 0.7)
        if dates[i].year == 2024:
            shock += rng.normal(0.4, 1.0)

        infl = max(0.4, 0.65 * infl + 0.55 + shock)
        depreciation = max(0.1, infl * 0.9 + rng.normal(0.3, 0.6))

        fx *= (1 + depreciation / 100)
        rate = max(15, 20 + infl * 5 + rng.normal(0, 5))
        mb *= (1 + max(0.1, infl + rng.normal(0.6, 1.2)) / 100)
        res += rng.normal(-120, 900)

        # Simulated REM expectation: informed but imperfect
        rem = max(0.2, infl + rng.normal(0, 0.45))

        inflation.append(infl)
        fx_rate.append(fx)
        policy_rate.append(rate)
        monetary_base.append(mb)
        reserves.append(res)
        rem_expectation.append(rem)

    return pd.DataFrame({
        "date": dates,
        "inflation_mom": inflation,
        "rem_expectation": rem_expectation,
        "fx_rate": fx_rate,
        "policy_rate": policy_rate,
        "monetary_base": monetary_base,
        "reserves": reserves
    })


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = (
        df.columns
        .str.strip()
        .str.lower()
        .str.replace(" ", "_")
        .str.replace("-", "_")
        .str.replace(".", "_", regex=False)
    )
    return df


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Builds lagged and transformed variables.
    Target is next-month inflation.
    """
    df = df.copy()
    df = df.sort_values("date")

    # Main inflation lags
    for lag in [1, 2, 3, 6, 12]:
        df[f"inflation_lag_{lag}"] = df["inflation_mom"].shift(lag)

    # Macro monthly percentage changes
    if "fx_rate" in df.columns:
        df["fx_depreciation_mom"] = df["fx_rate"].pct_change() * 100

    if "monetary_base" in df.columns:
        df["monetary_base_growth_mom"] = df["monetary_base"].pct_change() * 100

    if "reserves" in df.columns:
        df["reserves_change_mom"] = df["reserves"].pct_change() * 100

    # Calendar effect
    df["month"] = df["date"].dt.month

    # Forecasting target: next month inflation
    df["target_next_inflation"] = df["inflation_mom"].shift(-1)

    return df


def time_train_test_split(df: pd.DataFrame, test_size: float):
    """
    Time-series split: first part train, last part test.
    """
    split_idx = int(len(df) * (1 - test_size))
    train = df.iloc[:split_idx].copy()
    test = df.iloc[split_idx:].copy()
    return train, test


def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))


def fit_predict_models(train, test, features):
    X_train = train[features]
    y_train = train["target_next_inflation"]
    X_test = test[features]
    y_test = test["target_next_inflation"]

    models = {
        "Naive: last inflation": None,
        "Linear Regression": LinearRegression(),
        "Ridge Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("model", Ridge(alpha=1.0))
        ]),
        "Random Forest": RandomForestRegressor(
            n_estimators=400,
            max_depth=4,
            min_samples_leaf=4,
            random_state=42
        )
    }

    predictions = pd.DataFrame({
        "date": test["date"].values,
        "actual": y_test.values
    })

    metrics = []

    # Naive forecast: next month equals current month inflation
    naive_pred = test["inflation_mom"].values
    predictions["Naive: last inflation"] = naive_pred

    metrics.append({
        "model": "Naive: last inflation",
        "MAE": mean_absolute_error(y_test, naive_pred),
        "RMSE": rmse(y_test, naive_pred),
        "Bias": np.mean(naive_pred - y_test)
    })

    for name, model in models.items():
        if model is None:
            continue

        model.fit(X_train, y_train)
        pred = model.predict(X_test)

        predictions[name] = pred
        metrics.append({
            "model": name,
            "MAE": mean_absolute_error(y_test, pred),
            "RMSE": rmse(y_test, pred),
            "Bias": np.mean(pred - y_test)
        })

    # REM benchmark, if available
    if "rem_expectation" in test.columns:
        rem_test = test.dropna(subset=["rem_expectation", "target_next_inflation"])
        if len(rem_test) > 0:
            rem_pred = rem_test["rem_expectation"].values
            rem_actual = rem_test["target_next_inflation"].values

            predictions.loc[predictions["date"].isin(rem_test["date"]), "BCRA REM"] = rem_pred

            metrics.append({
                "model": "BCRA REM",
                "MAE": mean_absolute_error(rem_actual, rem_pred),
                "RMSE": rmse(rem_actual, rem_pred),
                "Bias": np.mean(rem_pred - rem_actual)
            })

    metrics_df = pd.DataFrame(metrics).sort_values("MAE")
    return predictions, metrics_df, models


def next_month_forecast(df_model, features):
    """
    Fits models on all available data and forecasts the next month.
    """
    df_model = df_model.dropna(subset=features + ["target_next_inflation"]).copy()

    latest_row = df_model.iloc[[-1]][features]
    train = df_model.iloc[:-1].copy()

    X = train[features]
    y = train["target_next_inflation"]

    final_models = {
        "Linear Regression": LinearRegression(),
        "Ridge Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("model", Ridge(alpha=1.0))
        ]),
        "Random Forest": RandomForestRegressor(
            n_estimators=400,
            max_depth=4,
            min_samples_leaf=4,
            random_state=42
        )
    }

    forecasts = []
    for name, model in final_models.items():
        model.fit(X, y)
        pred = model.predict(latest_row)[0]
        forecasts.append({"model": name, "forecast_next_month": pred})

    # Naive
    forecasts.append({
        "model": "Naive: last inflation",
        "forecast_next_month": df_model.iloc[-1]["inflation_mom"]
    })

    if "rem_expectation" in df_model.columns:
        last_rem = df_model.iloc[-1].get("rem_expectation", np.nan)
        if pd.notna(last_rem):
            forecasts.append({
                "model": "BCRA REM",
                "forecast_next_month": last_rem
            })

    return pd.DataFrame(forecasts).sort_values("forecast_next_month")


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.header("1. Data input")

uploaded_file = st.sidebar.file_uploader(
    "Upload your monthly CSV",
    type=["csv"]
)

use_demo = st.sidebar.checkbox(
    "Use simulated demo data",
    value=True if uploaded_file is None else False
)

st.sidebar.markdown(
    """
Expected minimum columns:

- `date`
- `inflation_mom`

Optional columns:

- `rem_expectation`
- `fx_rate`
- `policy_rate`
- `monetary_base`
- `reserves`
"""
)

test_size = st.sidebar.slider(
    "Test set size",
    min_value=0.10,
    max_value=0.40,
    value=0.25,
    step=0.05
)

st.sidebar.header("2. Model options")
include_rem = st.sidebar.checkbox("Use REM as benchmark", value=True)
include_fx = st.sidebar.checkbox("Use exchange rate", value=True)
include_policy_rate = st.sidebar.checkbox("Use policy/interest rate", value=True)
include_money = st.sidebar.checkbox("Use monetary base", value=True)
include_reserves = st.sidebar.checkbox("Use reserves", value=True)


# ============================================================
# LOAD DATA
# ============================================================

if uploaded_file is not None:
    raw_df = pd.read_csv(uploaded_file)
    raw_df = normalize_columns(raw_df)
else:
    raw_df = make_demo_data()
    raw_df = normalize_columns(raw_df)

# Validate basic columns
if "date" not in raw_df.columns or "inflation_mom" not in raw_df.columns:
    st.error("Your CSV must include at least `date` and `inflation_mom` columns.")
    st.stop()

raw_df["date"] = pd.to_datetime(raw_df["date"], errors="coerce")
raw_df = raw_df.dropna(subset=["date", "inflation_mom"]).sort_values("date")

st.info(
    "Demo data is simulated and should only be used to test the app. "
    "For the real project, upload official/processed monthly data."
)

with st.expander("Preview dataset"):
    st.dataframe(raw_df.tail(12), use_container_width=True)


# ============================================================
# FEATURE ENGINEERING
# ============================================================

df = add_features(raw_df)

base_features = [
    "inflation_lag_1",
    "inflation_lag_2",
    "inflation_lag_3",
    "inflation_lag_6",
    "inflation_lag_12",
    "month"
]

optional_features = []

if include_fx and "fx_depreciation_mom" in df.columns:
    optional_features.append("fx_depreciation_mom")

if include_policy_rate and "policy_rate" in df.columns:
    optional_features.append("policy_rate")

if include_money and "monetary_base_growth_mom" in df.columns:
    optional_features.append("monetary_base_growth_mom")

if include_reserves and "reserves_change_mom" in df.columns:
    optional_features.append("reserves_change_mom")

if include_rem and "rem_expectation" in df.columns:
    optional_features.append("rem_expectation")

features = base_features + optional_features

df_model = df.dropna(subset=features + ["target_next_inflation"]).copy()

if len(df_model) < 36:
    st.warning(
        "The model has fewer than 36 usable observations after creating lags. "
        "Results may be unstable."
    )

train, test = time_train_test_split(df_model, test_size=test_size)

# ============================================================
# MAIN DASHBOARD
# ============================================================

col1, col2, col3, col4 = st.columns(4)

latest_month = raw_df["date"].max().strftime("%Y-%m")
latest_inflation = raw_df.loc[raw_df["date"].idxmax(), "inflation_mom"]

col1.metric("Latest month", latest_month)
col2.metric("Latest monthly inflation", f"{latest_inflation:.2f}%")
col3.metric("Training observations", len(train))
col4.metric("Testing observations", len(test))

st.divider()

st.header("Inflation dynamics")

inflation_chart = alt.Chart(raw_df).mark_line(point=True).encode(
    x=alt.X("date:T", title="Date"),
    y=alt.Y("inflation_mom:Q", title="Monthly inflation (%)"),
    tooltip=[
        alt.Tooltip("date:T", title="Date"),
        alt.Tooltip("inflation_mom:Q", title="Monthly inflation", format=".2f")
    ]
).properties(height=350)

st.altair_chart(inflation_chart, use_container_width=True)


# ============================================================
# MODELING
# ============================================================

st.header("Model comparison")

predictions, metrics_df, models = fit_predict_models(train, test, features)

st.subheader("Forecast accuracy")
st.dataframe(
    metrics_df.style.format({
        "MAE": "{:.3f}",
        "RMSE": "{:.3f}",
        "Bias": "{:.3f}"
    }),
    use_container_width=True
)

best_model = metrics_df.iloc[0]["model"]
best_mae = metrics_df.iloc[0]["MAE"]

st.success(f"Best model by MAE: **{best_model}** with MAE of **{best_mae:.2f} percentage points**.")

# Long format for chart
pred_long = predictions.melt(
    id_vars=["date", "actual"],
    var_name="model",
    value_name="prediction"
).dropna()

actual_long = predictions[["date", "actual"]].rename(columns={"actual": "value"})
actual_long["series"] = "Actual inflation"

forecast_long = pred_long.rename(columns={"prediction": "value"})
forecast_long["series"] = forecast_long["model"]

chart_df = pd.concat([
    actual_long[["date", "value", "series"]],
    forecast_long[["date", "value", "series"]]
], ignore_index=True)

forecast_chart = alt.Chart(chart_df).mark_line(point=True).encode(
    x=alt.X("date:T", title="Date"),
    y=alt.Y("value:Q", title="Monthly inflation / forecast (%)"),
    color=alt.Color("series:N", title="Series"),
    tooltip=[
        alt.Tooltip("date:T", title="Date"),
        alt.Tooltip("series:N", title="Series"),
        alt.Tooltip("value:Q", title="Value", format=".2f")
    ]
).properties(height=420)

st.subheader("Actual vs predicted inflation")
st.altair_chart(forecast_chart, use_container_width=True)


# ============================================================
# NEXT MONTH FORECAST
# ============================================================

st.header("Next-month forecast")

forecast_df = next_month_forecast(df, features)

st.dataframe(
    forecast_df.style.format({"forecast_next_month": "{:.2f}%"}),
    use_container_width=True
)

avg_forecast = forecast_df["forecast_next_month"].mean()
min_forecast = forecast_df["forecast_next_month"].min()
max_forecast = forecast_df["forecast_next_month"].max()

c1, c2, c3 = st.columns(3)
c1.metric("Average forecast", f"{avg_forecast:.2f}%")
c2.metric("Minimum forecast", f"{min_forecast:.2f}%")
c3.metric("Maximum forecast", f"{max_forecast:.2f}%")

st.markdown(
    """
**Interpretation tip:**  
If the Python models forecast meaningfully above the REM, the model is detecting more inflationary pressure than the market consensus.
If the models forecast below REM, the market may be pricing in risks that are not fully captured by your selected variables.
"""
)


# ============================================================
# FEATURE IMPORTANCE
# ============================================================

st.header("Feature importance")

if len(train) > 20:
    rf = RandomForestRegressor(
        n_estimators=400,
        max_depth=4,
        min_samples_leaf=4,
        random_state=42
    )

    rf.fit(train[features], train["target_next_inflation"])

    importance_df = pd.DataFrame({
        "feature": features,
        "importance": rf.feature_importances_
    }).sort_values("importance", ascending=False)

    importance_chart = alt.Chart(importance_df).mark_bar().encode(
        x=alt.X("importance:Q", title="Importance"),
        y=alt.Y("feature:N", sort="-x", title="Feature"),
        tooltip=[
            alt.Tooltip("feature:N", title="Feature"),
            alt.Tooltip("importance:Q", title="Importance", format=".3f")
        ]
    ).properties(height=350)

    st.altair_chart(importance_chart, use_container_width=True)

    with st.expander("Feature importance table"):
        st.dataframe(importance_df, use_container_width=True)
else:
    st.warning("Not enough training observations to estimate feature importance reliably.")


# ============================================================
# PROJECT NOTES
# ============================================================

st.divider()

st.header("Methodological notes")

st.markdown(
    """
- The target is **next-month monthly inflation**.
- The train/test split respects time order. The model is trained on the past and tested on later months.
- The REM is treated as a benchmark forecast, not as an explanatory variable unless you select it in the sidebar.
- The naive benchmark assumes next month inflation equals the latest observed monthly inflation.
- MAE is easy to interpret: an MAE of 0.50 means the model is wrong by 0.50 percentage points on average.
- Random Forest can capture nonlinear behavior, but it is less interpretable than linear regression.
"""
)

st.warning(
    "This dashboard is a forecasting experiment. It is not investment, policy, or financial advice."
)
