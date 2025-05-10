import pandas as pd
import numpy as np
import datetime
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

# helper to parse times
def to_seconds(val):
    if pd.isna(val):
        return np.nan
    if isinstance(val, pd.Timedelta):
        return val.total_seconds()
    if isinstance(val, datetime.time):
        return val.minute * 60 + val.second + val.microsecond / 1_000_000
    if isinstance(val, str):
        try:
            return pd.to_timedelta(val).total_seconds()
        except:
            mins, secs = val.split(":")
            return int(mins) * 60 + float(secs)
    return float(val)

# load data
lap_df = pd.read_excel(
    "\2024SonsioGPLapChart.xlsx"
)
results_df = pd.read_excel(
    "2024SonsioGPRaceResults.xlsx"
)
qual_df = pd.read_excel(
    "2025SonsioGPQualifying.xlsx"
)

# pivot lap chart
lap_long = lap_df.melt(
    id_vars="DRIVER",
    var_name="Lap",
    value_name="LapTime"
).dropna(subset=["LapTime"])

# convert to seconds
lap_long["LapTime_s"] = lap_long["LapTime"].apply(to_seconds)

# calculate
avg_lap = (
    lap_long
    .groupby("DRIVER", as_index=False)["LapTime_s"]
    .mean()
    .rename(columns={"DRIVER": "Driver", "LapTime_s": "AvgLapTime"})
)

# quali times to seconds
qual_df["QualifyingTime_s"] = qual_df["Time"].apply(to_seconds)

# merge
data = pd.merge(
    avg_lap,
    qual_df[["Driver", "QualifyingTime_s"]],
    on="Driver",
    how="inner"
)

# dataframe
print("\n=== Data used for modeling ===")
print(data.to_string(index=False))

# define features and target
X = data[["QualifyingTime_s"]].values.reshape(-1, 1)
y = data["AvgLapTime"].values

# train
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
linreg = LinearRegression()
linreg.fit(X_train, y_train)

# eval
y_pred_test = linreg.predict(X_test)
print(f"\n🔍 Linear Model MAE: {mean_absolute_error(y_test, y_pred_test):.2f} seconds")

# prediction
data["PredAvgLapTime_lin"] = linreg.predict(data[["QualifyingTime_s"]])
ranking = data.sort_values("PredAvgLapTime_lin").reset_index(drop=True)

print("\n🏁 Predicted Sonsio GP Podium (linear model, fastest avg lap) 🏁")
print(ranking[["Driver", "PredAvgLapTime_lin"]].head(3).to_string(index=False))
print(f"\n🏆 Predicted Winner: {ranking.iloc[0]['Driver']}")
