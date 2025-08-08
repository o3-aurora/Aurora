import os
import json
import joblib
import pandas as pd
import numpy as np
import tensorflow as tf

# Load settings
model = tf.keras.models.load_model("model.keras", compile=False)
scaler = joblib.load("scaler.pkl")
with open("metadata.json") as f:
    meta = json.load(f)

features = meta["features"]
window = meta["window"]
threshold = 0.53 # Decision threshold for bull/bear classification

# Load hourly CSV from ENV
filepath = os.getenv("DATA")
if not filepath or not os.path.exists(filepath):
    raise RuntimeError("DATA environment variable not set or file not found.")

df = pd.read_csv(filepath, parse_dates=["date"])
df.sort_values("date", inplace=True)

# Calculate actual momentum labels on the close price
def labelmomentum(close, emaperiod=10, threshold=0.005):
    ema = pd.Series(close).ewm(span=emaperiod).mean()
    diff = (close - ema) / ema
    labels = (diff > threshold).astype(int)
    return labels.values

df['momentumlabel'] = labelmomentum(df['close'].values)
df.dropna(inplace=True)

# Reverse dataframe: newest entry first
df = df.iloc[::-1].reset_index(drop=True) # Newest at index 0

# Keep only full days (24h chunks) from newest
numrows = len(df)
fulldayscount = numrows // 24
df = df.iloc[:fulldayscount * 24] # Truncate to full days only

# Limit to last 24 days max
fulldayscount = min(fulldayscount, 24)

# Hour from newest entry for date labels
lasthour = df.iloc[0]['date'].hour

# Predict & check correctness
predictions = []
corrects = []
for dayidx in range(fulldayscount):
    daystart = dayidx * 24
    dayend = daystart + 24

    if dayend < window:
        continue

    xwindow = df.iloc[dayend - window:dayend][features].values
    xscaled = scaler.transform(xwindow).reshape(1, window, len(features))

    prob = model.predict(xscaled, verbose=0).item()
    predlabel = int(prob >= threshold)
    predstate = "Bull" if predlabel == 1 else "Bear"

    actuallabel = df.iloc[dayend - 1]['momentumlabel']
    correct = 1 if predlabel == actuallabel else 0
    corrects.append(correct)

    # Shift date forward by 1 day to align with last data date
    dateonly = pd.to_datetime(df.iloc[dayend - 1]['date']).normalize()
    labeleddate = dateonly + pd.Timedelta(days=1) + pd.Timedelta(hours=lasthour)

    predictions.append((labeleddate, predstate, prob, correct))

# Sort historical predictions ascending by date
historicalpreds = sorted(predictions, key=lambda x: x[0])

# Next day prediction: day after last historical prediction (use newest historical date)
lastpreddate = historicalpreds[-1][0]

lastwindowdata = df[features].iloc[:window-1].values
lastrow = df[features].iloc[[0]].values # Newest row
nextdayinput = np.vstack([lastwindowdata, lastrow])

nextdayscaled = scaler.transform(nextdayinput).reshape(1, window, len(features))
nextprob = model.predict(nextdayscaled, verbose=0).item()
nextlabel = int(nextprob >= threshold)
nextstate = "Bull" if nextlabel == 1 else "Bear"

nextdaydate = lastpreddate + pd.Timedelta(days=1)
predictions.append((nextdaydate, nextstate, nextprob, "-"))
futurepred = predictions[-1]

# Print historical predictions oldest to newest
for date, state, prob, correct in historicalpreds:
    print(f"{date}: {state} (prob={prob:.4f}) Correct? {correct}")

# Print next day forecast last
date, state, prob, correct = futurepred
print(f"{date}: {state} (prob={prob:.4f}) Correct? {correct}")

# Print accuracy percentage
if corrects:
    accuracy = sum(corrects) / len(corrects) * 100
    print(f"\nHistorical Accuracy: {accuracy:.2f}% ({sum(corrects)}/{len(corrects)})")
else:
    print("\nHistorical Accuracy: No historical predictions to calculate accuracy.")
