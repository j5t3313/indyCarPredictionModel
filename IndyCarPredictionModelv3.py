import pandas as pd
import numpy as np
import datetime
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error

def to_seconds(val):
    """Turn strings like '01:12.3', datetime.time, or Timedelta → float seconds."""
    if pd.isna(val):
        return np.nan
    if isinstance(val, pd.Timedelta):
        return val.total_seconds()
    if isinstance(val, datetime.time):
        return val.minute*60 + val.second + val.microsecond/1e6
    if isinstance(val, str):
        try:
            return pd.to_timedelta(val).total_seconds()
        except:
            mins, secs = val.split(":")
            return int(mins)*60 + float(secs)
    return float(val)

# load data
lap_df     = pd.read_excel("\2024SonsioGPLapChart.xlsx")
results_df = pd.read_excel("\2024SonsioGPRaceResults.xlsx")
qual_df    = pd.read_excel("\2025SonsioGPQualifying.xlsx")

# pivot lapchart and parse timings
lap_long = (
    lap_df
    .melt(id_vars="DRIVER", var_name="Lap", value_name="LapTime")
    .dropna(subset=["LapTime"])
)
lap_long["LapTime_s"] = lap_long["LapTime"].apply(to_seconds)

# compute driver stats from 2024 Sonsio GP
stats = (
    lap_long
    .groupby("DRIVER")["LapTime_s"]
    .agg(["mean","std","min"])
    .reset_index()
    .rename(columns={
        "DRIVER":"Driver",
        "mean":"LapTimeMean",
        "std": "LapTimeStd",
        "min": "LapTimeMin"
    })
)

# grab finish pos
race_pos = results_df[["Driver","Pos"]].rename(columns={"Pos":"FinishPos"})

# parse quali times
qual_df["QualifyingTime_s"] = qual_df["Time"].apply(to_seconds)

# merge dataframe
data = (
    stats
    .merge(race_pos,      on="Driver")
    .merge(qual_df[["Driver","QualifyingTime_s"]], on="Driver")
)

print("\n=== Full Data for Modeling ===")
print(data.to_string(index=False))

# build features
feature_cols = ["QualifyingTime_s","LapTimeStd","LapTimeMin","FinishPos"]
X = data[feature_cols].values
y = data["LapTimeMean"].values

# split, train, evaluate
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
model = GradientBoostingRegressor(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=4,
    random_state=42
)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

# predict average laptimes and rank drivers
data["PredAvgLapTime"] = model.predict(data[feature_cols])
ranking = data.sort_values("PredAvgLapTime").reset_index(drop=True)

print("\n🏁 Predicted Sonsio GP Podium (fastest avg lap) 🏁")
print(ranking[["Driver","PredAvgLapTime"]].head(3).to_string(index=False))
print(f"\n🏆 Predicted Winner: {ranking.iloc[0]['Driver']}")
print(f"\n🔍 Model MAE: {mean_absolute_error(y_test, y_pred):.2f} seconds")