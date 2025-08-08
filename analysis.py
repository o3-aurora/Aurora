import os
import pandas as pd
import numpy as np
import tensorflow as tf
import random
import json
import joblib

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.metrics import Precision, Recall, AUC
import tensorflow.keras.backend as K

# Reproducibility
seed = 42
random.seed(seed)
np.random.seed(seed)
tf.random.set_seed(seed)

# Load CSV
filepath = os.getenv('DATA')
if not filepath or not os.path.exists(filepath):
    raise RuntimeError("DATA environment variable is not set or file not found.")
df = pd.read_csv(filepath, parse_dates=['date'])
df.sort_values('date', inplace=True)
df.reset_index(drop=True, inplace=True)

# Required features
features = ['open', 'high', 'low', 'close', 'volume', 'SMA', 'lower', 'upper',
            'EMA', 'RSI', 'histogram', 'MACD', 'signal', 'VWAP']
for col in features:
    if col not in df.columns:
        raise KeyError(f"Missing required column: {col}")
data = df[features].copy()

# Momentum Regime Labeling
def labelmomentum(close, emaperiod=10, threshold=0.005):
    ema = pd.Series(close).ewm(span=emaperiod).mean()
    diff = (close - ema) / ema
    labels = (diff > threshold).astype(int)
    return labels.values

df['momentumlabel'] = labelmomentum(df['close'].values)
df.dropna(inplace=True)

# Align
labels = df['momentumlabel'].values
if len(df) <= 24:
    raise ValueError("Not enough rows for a 24-step window.")
df = df.iloc[24:]
labels = labels[24:]
print("Momentum regime distribution (0=bear, 1=bull):", np.bincount(labels))

# Sliding windows
window = 24
x, y = [], []
for i in range(len(df) - window):
    x.append(df[features].iloc[i:i+window].values)
    y.append(labels[i + window - 1])
x = np.array(x)
y = np.array(y)

# Train/test split
split = int(len(x) * 0.8)
xtrain, xtest = x[:split], x[split:]
ytrain, ytest = y[:split], y[split:]

# Scale using training data only
scaler = StandardScaler()
xtrainreshaped = xtrain.reshape(-1, len(features))
xtestreshaped = xtest.reshape(-1, len(features))
scaler.fit(xtrainreshaped)
xtrainscaled = scaler.transform(xtrainreshaped).reshape(xtrain.shape)
xtestscaled = scaler.transform(xtestreshaped).reshape(xtest.shape)

# Data augmentation with noise injection
xtrain_noisy = xtrainscaled + np.random.normal(0, 0.01, xtrainscaled.shape)
xtrain_combined = np.concatenate([xtrainscaled, xtrain_noisy])
ytrain_combined = np.concatenate([ytrain, ytrain])

# Class weights
if len(np.unique(ytrain_combined)) < 2:
    raise ValueError("Training labels contain only one class; cannot compute class weights.")
weights = compute_class_weight('balanced', classes=np.unique(ytrain_combined), y=ytrain_combined)
classweights = dict(enumerate(weights))
print("Class weights:", classweights)

# Custom focal loss function
def focalloss(gamma=2., alpha=0.25):
    def lossfn(y_true, y_pred):
        y_true = K.cast(y_true, dtype='float32')
        y_pred = K.clip(y_pred, K.epsilon(), 1. - K.epsilon())
        p_t = y_true * y_pred + (1 - y_true) * (1 - y_pred)
        alpha_t = y_true * alpha + (1 - y_true) * (1 - alpha)
        loss = -alpha_t * K.pow(1. - p_t, gamma) * K.log(p_t)
        return K.mean(loss)
    return lossfn

# Model architecture with custom focal loss
def buildmodel():
    inp = tf.keras.Input(shape=(window, len(features)))
    x = tf.keras.layers.Conv1D(32, 3, activation='relu', padding='same')(inp)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPooling1D(2)(x)
    x = tf.keras.layers.GRU(64)(x)
    x = tf.keras.layers.Dense(32, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    out = tf.keras.layers.Dense(1, activation='sigmoid')(x)
    model = tf.keras.Model(inp, out)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss=focalloss(gamma=2.0, alpha=0.25),
        metrics=['accuracy', Precision(), Recall(), AUC()]
    )
    return model

# Training
model = buildmodel()
earlystop = EarlyStopping(monitor='val_auc', patience=10, restore_best_weights=True, mode='max')
reduce_lr = ReduceLROnPlateau(monitor='val_auc', factor=0.5, patience=2, verbose=1, mode='max')

model.fit(
    xtrain_combined, ytrain_combined,
    validation_data=(xtestscaled, ytest),
    epochs=100,
    batch_size=96,
    callbacks=[earlystop, reduce_lr],
    class_weight=classweights,
    verbose=2
)

# Evaluation using optimized threshold
yprob = model.predict(xtestscaled).flatten()
yopt = (yprob >= 0.53).astype(int)
acc = (yopt == ytest).mean()
f1 = f1_score(ytest, yopt, zero_division=0)
print(f"\nFinal Test Accuracy: {acc:.4f}")
print(f"Final Test F1 Score (Threshold=0.53): {f1:.4f}")

# Save model
model.save("model.keras")

# Save scaler
joblib.dump(scaler, 'scaler.pkl')

# Save metadata
metadata = {
    "features": features,
    "window": window,
    "threshold": 0.005,
    "class_weights": classweights
}
with open("metadata.json", "w") as f:
    json.dump(metadata, f, indent=2)
