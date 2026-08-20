#Transit Delay Analysis
# This script looks at transit delays and builds a regression model.

import json
import pickle
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "public_transport_delays.csv"
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

sns.set_theme(style="whitegrid")

# columns used for the model
CATEGORICAL_COLS = ["weather_condition", "transport_type", "season", "weekday_name"]
NUMERIC_COLS = ["temperature_C", "precipitation_mm", "wind_speed_kmh",
                 "traffic_congestion_index", "peak_hour", "holiday", "has_event"]
TARGET_COL = "actual_arrival_delay_min"


def load_data():
    df = pd.read_csv(DATA_PATH)
    df["event_type"] = df["event_type"].fillna("None")
    df["has_event"] = (df["event_type"] != "None").astype(int)
    df["hour"] = pd.to_datetime(df["time"], format="%H:%M:%S").dt.hour
    weekday_map = {0: "Mon", 1: "Tue", 2: "Wed", 3: "Thu", 4: "Fri", 5: "Sat", 6: "Sun"}
    df["weekday_name"] = df["weekday"].map(weekday_map)
    return df


def descriptive_stats(df):
    # basic summary of the data
    stats_summary = {
        "n_rows": len(df),
        "mean_delay": round(df["actual_arrival_delay_min"].mean(), 2),
        "median_delay": round(df["actual_arrival_delay_min"].median(), 2),
        "std_delay": round(df["actual_arrival_delay_min"].std(), 2),
        "pct_delayed": round(df["delayed"].mean() * 100, 1),
        "mean_delay_by_transport": df.groupby("transport_type")["actual_arrival_delay_min"].mean().round(2).to_dict(),
        "mean_delay_by_weather": df.groupby("weather_condition")["actual_arrival_delay_min"].mean().round(2).to_dict(),
        "mean_delay_by_peak": df.groupby("peak_hour")["actual_arrival_delay_min"].mean().round(2).to_dict(),
        "mean_delay_by_weekday": df.groupby("weekday_name")["actual_arrival_delay_min"].mean().round(2).to_dict(),
    }
    return stats_summary


def hypothesis_tests(df):
    num_tests = 5
    alpha = 0.05
    alpha_corrected = alpha / num_tests

    results = {}

    groups = [g["actual_arrival_delay_min"].values for _, g in df.groupby("weather_condition")]
    f_stat, p_val = stats.f_oneway(*groups)
    results["anova_weather"] = {
        "test": "One-way ANOVA: delay ~ weather_condition",
        "f_stat": round(float(f_stat), 3),
        "p_value": round(float(p_val), 5),
        "significant_at_bonferroni_alpha": bool(p_val < alpha_corrected),
    }

    peak = df.loc[df["peak_hour"] == 1, "actual_arrival_delay_min"]
    offpeak = df.loc[df["peak_hour"] == 0, "actual_arrival_delay_min"]
    t_stat, p_val = stats.ttest_ind(peak, offpeak, equal_var=False)
    results["ttest_peak_hour"] = {
        "test": "Welch's t-test: delay (peak hour) vs delay (off-peak)",
        "t_stat": round(float(t_stat), 3),
        "p_value": round(float(p_val), 5),
        "significant_at_bonferroni_alpha": bool(p_val < alpha_corrected),
        "mean_peak": round(float(peak.mean()), 2),
        "mean_offpeak": round(float(offpeak.mean()), 2),
    }

    groups = [g["actual_arrival_delay_min"].values for _, g in df.groupby("weekday_name")]
    f_stat, p_val = stats.f_oneway(*groups)
    results["anova_weekday"] = {
        "test": "One-way ANOVA: delay ~ day_of_week",
        "f_stat": round(float(f_stat), 3),
        "p_value": round(float(p_val), 5),
        "significant_at_bonferroni_alpha": bool(p_val < alpha_corrected),
    }

    groups = [g["actual_arrival_delay_min"].values for _, g in df.groupby("transport_type")]
    f_stat, p_val = stats.f_oneway(*groups)
    results["anova_transport_type"] = {
        "test": "One-way ANOVA: delay ~ transport_type",
        "f_stat": round(float(f_stat), 3),
        "p_value": round(float(p_val), 5),
        "significant_at_bonferroni_alpha": bool(p_val < alpha_corrected),
    }

    r, p_val = stats.pearsonr(df["traffic_congestion_index"], df["actual_arrival_delay_min"])
    results["corr_traffic_congestion"] = {
        "test": "Pearson correlation: traffic_congestion_index vs delay",
        "r": round(float(r), 3),
        "p_value": round(float(p_val), 5),
        "significant_at_bonferroni_alpha": bool(p_val < alpha_corrected),
    }

    results["_note"] = f"Using Bonferroni-corrected alpha = {alpha}/{num_tests} = {alpha_corrected} since 5 tests were run"
    return results


def build_x(df, train_columns=None):
    
    x_num = df[NUMERIC_COLS].reset_index(drop=True)
    x_cat = pd.get_dummies(df[CATEGORICAL_COLS].reset_index(drop=True), drop_first=True)
    x = pd.concat([x_num, x_cat], axis=1)
    x = sm.add_constant(x, has_constant="add")
    x = x.astype(float)
    if train_columns is not None:
        x = x.reindex(columns=train_columns, fill_value=0.0)
    return x


def fit_model(df):
    # Splitting into train/test so I can actually check how well the model
    # does on data it hasn't seen, instead of just looking at R^2 on the
    # same rows it was fit on (which was a mistake in my first attempt --
    # that R^2 doesn't tell you anything about real predictive power).
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

    x_train = build_x(train_df)
    y_train = train_df[TARGET_COL].astype(float).reset_index(drop=True)

    model = sm.OLS(y_train, x_train).fit()

    x_test = build_x(test_df, train_columns=x_train.columns.tolist())
    y_test = test_df[TARGET_COL].astype(float).reset_index(drop=True)
    y_pred_test = model.predict(x_test)

    test_r2 = r2_score(y_test, y_pred_test)
    test_mae = mean_absolute_error(y_test, y_pred_test)

    return model, x_train.columns.tolist(), test_r2, test_mae


def main():
    df = load_data()

    desc = descriptive_stats(df)
    with open(OUTPUT_DIR / "descriptive_stats.json", "w") as f:
        json.dump(desc, f, indent=2)

    tests = hypothesis_tests(df)
    with open(OUTPUT_DIR / "hypothesis_tests.json", "w") as f:
        json.dump(tests, f, indent=2)

    model, train_columns, test_r2, test_mae = fit_model(df)

    with open(OUTPUT_DIR / "model_summary.txt", "w") as f:
        f.write("Design matrix: dummy-encoded categoricals (drop_first=True) + numeric predictors\n")
        f.write(f"Predictors: {CATEGORICAL_COLS + NUMERIC_COLS}\n")
        f.write("Reference (baseline) category for each dummy-encoded variable is whichever\n")
        f.write("category got dropped -- pandas drops the first one alphabetically by default.\n")
        f.write(f"Test set R^2 (held-out 20% of data, NOT the same rows the model was fit on): {test_r2:.4f}\n")
        f.write(f"Test set MAE: {test_mae:.2f} minutes\n\n")
        f.write("The model results should be interpreted carefully.\n\n")
        f.write(str(model.summary()))

    with open(OUTPUT_DIR / "model.pkl", "wb") as f:
        pickle.dump(model, f)

    categories = {
        "transport_type": sorted(df["transport_type"].unique().tolist()),
        "weather_condition": sorted(df["weather_condition"].unique().tolist()),
        "season": sorted(df["season"].unique().tolist()),
        "weekday_name": ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"],
        "route_id": sorted(df["route_id"].unique().tolist()),
        "train_columns": train_columns,
        "categorical_cols": CATEGORICAL_COLS,
        "numeric_cols": NUMERIC_COLS,
        "test_r2": round(float(test_r2), 4),
        "test_mae": round(float(test_mae), 2),
    }
    with open(OUTPUT_DIR / "categories.json", "w") as f:
        json.dump(categories, f, indent=2)

    # a few EDA plots
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))

    sns.boxplot(data=df, x="weather_condition", y="actual_arrival_delay_min", ax=axes[0, 0])
    axes[0, 0].set_title("Delay by Weather Condition")
    axes[0, 0].tick_params(axis="x", rotation=30)

    sns.boxplot(data=df, x="peak_hour", y="actual_arrival_delay_min", ax=axes[0, 1])
    axes[0, 1].set_title("Delay: Peak Hour vs Off-Peak")
    axes[0, 1].set_xticks([0, 1])
    axes[0, 1].set_xticklabels(["Off-Peak", "Peak"])

    sns.boxplot(data=df, x="weekday_name", y="actual_arrival_delay_min",
                order=["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"], ax=axes[1, 0])
    axes[1, 0].set_title("Delay by Day of Week")

    sns.scatterplot(data=df, x="traffic_congestion_index", y="actual_arrival_delay_min",
                     alpha=0.4, ax=axes[1, 1])
    axes[1, 1].set_title("Delay vs Traffic Congestion Index")

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "eda_plots.png", dpi=150)
    plt.close()

    print("Done! Outputs saved to:", OUTPUT_DIR)
    print("\nHypothesis test results:")
    for k, v in tests.items():
        if k == "_note":
            continue
        sig = "SIGNIFICANT" if v["significant_at_bonferroni_alpha"] else "not significant"
        print(f"  {v['test']}: p = {v['p_value']}  ({sig})")
    print(f"\nTraining R^2: {model.rsquared:.4f}")
    print(f"Test R^2: {test_r2:.4f}")
    print(f"Test MAE: {test_mae:.2f} min")


if __name__ == "__main__":
    main()