# Transit Delay Predictor - Streamlit app
# Pick some trip conditions and get a predicted delay with a confidence
# interval. Loads the model that run_analysis.py already trained and saved.

import json
import pickle
from pathlib import Path

import pandas as pd
import streamlit as st
from PIL import Image

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "outputs"

st.set_page_config(page_title="Transit Delay Predictor", page_icon="🚌")

# load everything run_analysis.py saved
with open(OUTPUT_DIR / "model.pkl", "rb") as f:
    model = pickle.load(f)
with open(OUTPUT_DIR / "categories.json") as f:
    categories = json.load(f)
with open(OUTPUT_DIR / "hypothesis_tests.json") as f:
    tests = json.load(f)
with open(OUTPUT_DIR / "descriptive_stats.json") as f:
    desc = json.load(f)

st.title("🚌 Transit Delay Predictor")
st.write("Pick your trip conditions below and get an expected delay with a confidence interval.")

with st.expander("About this dataset (read this first)"):
    st.write(
        "This uses a Kaggle practice dataset that looks like it was randomly "
        "generated (some rows have seasons and temperatures that don't match "
        "up in real life). Because of that, none of my hypothesis tests came "
        "back significant, and the model's R^2 on held-out test data is "
        f"actually **{categories['test_r2']}** (basically 0, sometimes even "
        "negative, meaning it does not really predict delay better than just "
        "guessing the average). I'm keeping this dashboard working end-to-end "
        "anyway since the pipeline itself is correct -- it's ready to plug in "
        "real transit data later, see the README."
    )

st.subheader("Trip conditions")

col1, col2 = st.columns(2)
with col1:
    transport_type = st.selectbox("Transport type", categories["transport_type"])
    weekday_name = st.selectbox("Day of week", categories["weekday_name"])
    hour = st.slider("Hour of day", 0, 23, 8)
    weather_condition = st.selectbox("Weather condition", categories["weather_condition"])
    season = st.selectbox("Season", categories["season"])

with col2:
    temperature_C = st.slider("Temperature (C)", -10, 40, 20)
    precipitation_mm = st.slider("Precipitation (mm)", 0.0, 20.0, 0.0, step=0.5)
    wind_speed_kmh = st.slider("Wind speed (km/h)", 0, 60, 15)
    traffic_congestion_index = st.slider("Traffic congestion index", 0, 100, 50)
    holiday = st.checkbox("Holiday")
    has_event = st.checkbox("Nearby event")

# peak hour is just derived from whatever hour was picked
peak_hour = 1 if hour in (7, 8, 9, 16, 17, 18) else 0
st.caption(f"Peak hour flag: {'Yes' if peak_hour else 'No'} (set automatically from the hour above)")

# NOTE: I originally had a route dropdown here too, but route_id isn't
# actually one of the columns the model was trained on (there are 20+
# different route IDs and no real signal from any single one in this
# dataset), so picking a route wouldn't have changed the prediction at all.
# That was misleading, so I removed it instead of leaving a dropdown that
# silently does nothing. If real per-route data existed with an actual
# effect, this would be worth adding back in properly.

new_row = pd.DataFrame([{
    "transport_type": transport_type,
    "weather_condition": weather_condition,
    "season": season,
    "temperature_C": temperature_C,
    "precipitation_mm": precipitation_mm,
    "wind_speed_kmh": wind_speed_kmh,
    "traffic_congestion_index": traffic_congestion_index,
    "peak_hour": peak_hour,
    "weekday_name": weekday_name,
    "holiday": int(holiday),
    "has_event": int(has_event),
}])

cat_cols = categories["categorical_cols"]
num_cols = categories["numeric_cols"]
x_num = new_row[num_cols].reset_index(drop=True)
x_cat = pd.get_dummies(new_row[cat_cols])
x = pd.concat([x_num, x_cat], axis=1)
x.insert(0, "const", 1.0)
x = x.reindex(columns=categories["train_columns"], fill_value=0.0).astype(float)

pred = model.get_prediction(x).summary_frame(alpha=0.05)
point_estimate = float(pred["mean"].iloc[0])
ci_low, ci_high = float(pred["mean_ci_lower"].iloc[0]), float(pred["mean_ci_upper"].iloc[0])
pi_low, pi_high = float(pred["obs_ci_lower"].iloc[0]), float(pred["obs_ci_upper"].iloc[0])

# A delay can't really be negative in a way that's useful to show here
# (a bus that's "-3 minutes late" just means it was early), so I'm
# clipping anything below 0 up to 0 for display. The raw model output can
# go negative because OLS doesn't know delay has a floor -- that's a
# limitation of using plain linear regression for this kind of target.
point_estimate_display = max(point_estimate, 0)
ci_low_display = max(ci_low, 0)
pi_low_display = max(pi_low, 0)

st.subheader("Predicted delay")
st.metric("Expected delay", f"{point_estimate_display:.1f} min")
st.write(f"95% confidence interval (average delay): {ci_low_display:.1f} to {ci_high:.1f} min")
st.write(f"95% prediction interval (this one trip): {pi_low_display:.1f} to {pi_high:.1f} min")
if pi_low < 0:
    st.caption(
        f"(Raw model output actually went negative here, down to {pi_low:.1f} -- "
        "clipped to 0 above since a negative delay isn't meaningful. This is a "
        "known limitation of plain linear regression on this kind of target.)"
    )

st.divider()
st.subheader("Dataset overview")
c1, c2, c3 = st.columns(3)
c1.metric("Trips in dataset", desc["n_rows"])
c2.metric("Mean delay", f"{desc['mean_delay']} min")
c3.metric("% trips delayed", f"{desc['pct_delayed']}%")

st.divider()

tab1, tab2, tab3 = st.tabs(["EDA plots", "Hypothesis tests", "Full model summary"])

with tab1:
    img_path = OUTPUT_DIR / "eda_plots.png"
    if img_path.exists():
        st.image(Image.open(img_path), width="stretch")
    else:
        st.write("Run analysis/run_analysis.py first to generate this.")

with tab2:
    st.write(tests.get("_note", ""))
    for key, v in tests.items():
        if key == "_note":
            continue
        sig = v["significant_at_bonferroni_alpha"]
        icon = "significant" if sig else "not significant"
        st.write(f"**{v['test']}** -- {icon}")
        details = {k: val for k, val in v.items() if k not in ("test", "significant_at_bonferroni_alpha")}
        st.json(details)

with tab3:
    with open(OUTPUT_DIR / "model_summary.txt") as f:
        st.text(f.read())

st.divider()
st.caption("Built with pandas, statsmodels, scipy, and Streamlit.")
