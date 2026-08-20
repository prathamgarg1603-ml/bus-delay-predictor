# 🚌 Transit Delay Predictor

Regression + hypothesis testing on public transit delay data, shipped as an
interactive Streamlit dashboard: pick a route/transport type and conditions,
get an expected delay estimate with a confidence interval.

**[Live demo →](#deployment)** (add your Streamlit Cloud link here once deployed)

---

## ⚠️ Important note on the current dataset

This repo currently runs on a Kaggle dataset (`data/public_transport_delays.csv`,
[source](https://www.kaggle.com/datasets/khushikyad001/public-transport-delays-with-weather-and-events))
that appears to be **synthetically generated** rather than scraped from a real,
live transit system — some rows pair seasons with physically inconsistent
temperatures (e.g. "Summer" at -2.9°C), which real observed weather data
wouldn't do.

**Honest result on this dataset:** none of the hypothesis tests below came
back statistically significant (using a Bonferroni-corrected alpha of 0.01,
since I ran 5 tests), and on a held-out test set the regression model's R²
is basically 0 (sometimes even slightly negative — meaning it does no
better than just guessing the average delay every time). In plain terms —
in *this* dataset, weather, time of day, and day of week do not
meaningfully predict delay.

**This is still a legitimate, complete result to report.** The pipeline
(EDA → hypothesis testing → regression → dashboard) works end-to-end and
is easy to re-point at a real dataset (see
[Swapping in real data](#swapping-in-real-data) below).

### Fixes made after a first review pass

A few issues came up when I reviewed this more carefully and fixed them:
- **Train/test split added.** The first version of this fit and evaluated
  the model on the exact same rows, which doesn't tell you anything about
  real predictive power. Now it trains on 80% and reports R²/MAE on the
  other 20% it never saw.
- **Bonferroni correction for the 5 hypothesis tests.** Running 5 tests at
  the normal 0.05 cutoff actually gives a ~22% chance of a false positive
  somewhere, not 5%. Now using alpha = 0.01 per test instead.
- **Negative delay predictions are now clipped and flagged.** Plain linear
  regression doesn't know a delay can't go below 0, so it can output
  something like "-3 minutes." The dashboard now clips this to 0 for
  display and explicitly says when it happened.
- **Removed the route dropdown from the dashboard.** It looked interactive
  but `route_id` was never actually a variable the model was trained on, so
  picking a different route silently did nothing. That was misleading, so I
  took it out rather than leave a dropdown with no real effect.
- **Documented reference (baseline) categories** in the model summary —
  dummy encoding drops one category per variable as a baseline, and the
  first version never said which one, making the coefficients hard to
  interpret on their own.
- One thing I noticed but I'm **not** claiming as a finding: one individual
  coefficient (a weekday effect) showed p < 0.05 in the regression even
  though the overall model F-test is not significant. With ~24 predictors
  tested, that's very likely a false positive from multiple comparisons,
  not a real effect — noted in `model_summary.txt` instead of ignored.

---

## What this project does

1. **`analysis/run_analysis.py`** — loads the data, runs descriptive stats,
   runs 5 hypothesis tests, fits an OLS regression, and saves all outputs
   (model, plots, JSON summaries) to `outputs/`.
2. **`app.py`** — a Streamlit dashboard that loads those saved outputs and
   lets a user pick trip conditions to get a predicted delay with a 95%
   confidence interval (uncertainty in the average) and a 95% prediction
   interval (uncertainty for one trip).

## Methodology

- **Descriptive statistics**: mean/median/std delay overall and broken out
  by transport type, weather, peak hour, and weekday.
- **Hypothesis tests**:
  - One-way ANOVA: delay ~ weather condition
  - Welch's t-test: delay during peak hour vs. off-peak
  - One-way ANOVA: delay ~ day of week
  - One-way ANOVA: delay ~ transport type
  - Pearson correlation: traffic congestion index vs. delay
- **Regression model**: OLS with dummy-encoded categorical predictors
  (weather, transport type, season, weekday) and numeric predictors
  (temperature, precipitation, wind speed, traffic congestion index,
  peak hour, holiday, nearby event flag), predicting
  `actual_arrival_delay_min`.

All test statistics, p-values, and the full model summary are saved as
JSON/text in `outputs/` and are also viewable inside the dashboard itself
(Hypothesis Tests and Full Model Summary tabs).

## Running locally

```bash
git clone <your-repo-url>
cd bus-delay-predictor
pip install -r requirements.txt

# regenerate the model + stats (optional — outputs/ is already included)
python analysis/run_analysis.py

# launch the dashboard
streamlit run app.py
```

## Deployment

Deploy for free on [Streamlit Community Cloud](https://streamlit.io/cloud):
push this repo to GitHub, connect it at share.streamlit.io, point it at
`app.py`, and it will build from `requirements.txt` automatically.

## Swapping in real data

To re-run this pipeline on real campus shuttle or transit-agency data:

1. Replace `data/public_transport_delays.csv` with your real data, keeping
   (or renaming to match) these columns: `transport_type`,
   `weather_condition`, `season`, `temperature_C`, `precipitation_mm`,
   `wind_speed_kmh`, `traffic_congestion_index`, `peak_hour`, `weekday`,
   `holiday`, `event_type`, `actual_arrival_delay_min`, `time`.
2. If your column names differ, update `CATEGORICAL_COLS`, `NUMERIC_COLS`,
   and `load_data()` in `analysis/run_analysis.py` to match.
3. Re-run `python analysis/run_analysis.py` — this regenerates
   `outputs/model.pkl` and everything the dashboard reads.
4. Re-run `streamlit run app.py` — no changes needed to `app.py` itself as
   long as the column names line up.

## Project structure

```
bus-delay-predictor/
├── app.py                       # Streamlit dashboard
├── requirements.txt
├── data/
│   └── public_transport_delays.csv
├── analysis/
│   └── run_analysis.py          # EDA + hypothesis tests + regression
└── outputs/                     # generated by run_analysis.py
    ├── model.pkl
    ├── categories.json
    ├── descriptive_stats.json
    ├── hypothesis_tests.json
    ├── model_summary.txt
    └── eda_plots.png
```

## Draft email to professor

> Subject: A quick stats project on transit delay prediction
>
> Hi Professor [Name],
>
> For a side project I built a full regression + hypothesis-testing pipeline
> on transit delay data — testing whether weather, time of day, day of week,
> and traffic congestion predict delay, then shipping the model as an
> interactive dashboard. On the dataset I used, none of the hypothesis tests
> came back significant (p > 0.05 across weather, peak hour, day of week,
> and transport type), and the regression model had very low explanatory
> power (R² ≈ 0.01) — I believe the dataset itself was synthetically
> generated rather than real observed data, which the analysis surfaced
> pretty clearly. I'm planning to re-run the same pipeline on [real
> campus/local transit data] next to see if the result holds. Code and
> dashboard: [your GitHub / Streamlit link].
>
> [Your name]

*(Edit this once you've swapped in real data and gotten an actual finding —
this draft is written to honestly reflect the current null result.)*

## Stack

Python · pandas · SciPy · statsmodels · Streamlit · matplotlib/seaborn
