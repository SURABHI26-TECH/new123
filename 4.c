# ============================
# Practical 4: Anomaly Detection using Autoencoder (LSTM)
# ============================

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout, RepeatVector, TimeDistributed
import os

# ----------------------------
# 1. Load and prepare dataset
# ----------------------------
csv_path = '/content/GOOG.csv'

# If file not found, auto-download using yfinance
if not os.path.exists(csv_path):
    print("⚠️ GOOG.csv not found — downloading from Yahoo Finance...")
    !pip install yfinance -q
    import yfinance as yf
    data = yf.download('GOOG', start='2010-01-01', end='2020-01-01')
    data.to_csv(csv_path)
    print("✅ Download complete and saved as GOOG.csv")

df = pd.read_csv(csv_path)
print("✅ Loaded CSV shape:", df.shape)
display(df.head())

# Keep only Date & Close
df = df[['Date', 'Close']].copy()
df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values(by='Date')
print("\nColumns after selecting Date and Close:")
display(df.head())
df.info()

print("\n📅 Date range:", df['Date'].min(), "to", df['Date'].max())

# Split train/test
train = df.loc[df['Date'] <= '2017-12-24'].copy()
test  = df.loc[df['Date'] > '2017-12-24'].copy()
print("Train shape:", train.shape, "Test shape:", test.shape)

# ----------------------------
# 2. Scaling
# ----------------------------
scaler = StandardScaler()
scaler.fit(np.array(train['Close']).reshape(-1, 1))

train['Close'] = scaler.transform(np.array(train['Close']).reshape(-1, 1))
test['Close']  = scaler.transform(np.array(test['Close']).reshape(-1, 1))

plt.figure(figsize=(10, 4))
plt.plot(train['Date'], train['Close'], label='Scaled - Train')
plt.legend()
plt.title('Scaled Close Price (Training Set)')
plt.show()

# ----------------------------
# 3. Create sequences
# ----------------------------
TIME_STEPS = 30

def create_sequences(X, time_steps=TIME_STEPS):
    Xs = []
    for i in range(len(X) - time_steps):
        v = X.iloc[i:(i + time_steps)].values
        Xs.append(v)
    return np.array(Xs)

X_train = create_sequences(train[['Close']])
X_test  = create_sequences(test[['Close']])

print("Training input shape:", X_train.shape)
print("Testing  input shape:", X_test.shape)

# ----------------------------
# 4. Build LSTM Autoencoder
# ----------------------------
np.random.seed(21)
tf.random.set_seed(21)

model = Sequential([
    LSTM(128, activation='tanh', input_shape=(X_train.shape[1], X_train.shape[2])),
    Dropout(0.2),
    RepeatVector(X_train.shape[1]),
    LSTM(128, activation='tanh', return_sequences=True),
    Dropout(0.2),
    TimeDistributed(Dense(X_train.shape[2]))
])

model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001), loss='mse')
model.summary()

# ----------------------------
# 5. Train model
# ----------------------------
history = model.fit(
    X_train, X_train,
    epochs=50,
    batch_size=32,
    validation_split=0.1,
    callbacks=[keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)],
    shuffle=False,
    verbose=2
)

plt.figure(figsize=(8, 4))
plt.plot(history.history['loss'], label='Training loss')
plt.plot(history.history['val_loss'], label='Validation loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('Training and Validation Loss')
plt.show()

# ----------------------------
# 6. Evaluate reconstruction error
# ----------------------------
X_train_pred = model.predict(X_train)
train_mae_loss = np.mean(np.abs(X_train_pred - X_train), axis=(1, 2))

plt.figure(figsize=(8, 4))
plt.hist(train_mae_loss, bins=50)
plt.xlabel('Train MAE loss')
plt.ylabel('Number of samples')
plt.title('Histogram of Training Reconstruction Error')
plt.show()

threshold = np.max(train_mae_loss)
print('Reconstruction error threshold (max):', threshold)

# ----------------------------
# 7. Test set anomaly detection
# ----------------------------
X_test_pred = model.predict(X_test, verbose=1)
test_mae_loss = np.mean(np.abs(X_test_pred - X_test), axis=(1, 2))

plt.figure(figsize=(8, 4))
plt.hist(test_mae_loss, bins=50)
plt.xlabel('Test MAE loss')
plt.ylabel('Number of samples')
plt.title('Histogram of Test Reconstruction Error')
plt.show()

anomaly_df = test[TIME_STEPS:].copy().reset_index(drop=True)
anomaly_df['loss'] = test_mae_loss
anomaly_df['threshold'] = threshold
anomaly_df['anomaly'] = anomaly_df['loss'] > anomaly_df['threshold']

anomalies = anomaly_df.loc[anomaly_df['anomaly']]
print("\n🚨 Number of anomalies detected:", anomalies.shape[0])
display(anomalies.head())

# ----------------------------
# 8. Plot anomalies
# ----------------------------
fig = go.Figure()
fig.add_trace(go.Scatter(
    x=anomaly_df['Date'],
    y=scaler.inverse_transform(anomaly_df['Close'].values.reshape(-1, 1)).flatten(),
    name='Close price'
))
fig.add_trace(go.Scatter(
    x=anomalies['Date'],
    y=scaler.inverse_transform(anomalies['Close'].values.reshape(-1, 1)).flatten(),
    mode='markers',
    name='Anomaly',
    marker=dict(color='red', size=6)
))
fig.update_layout(title='Detected Anomalies (LSTM Autoencoder)', showlegend=True)
fig.show()

# ----------------------------
# 9. Save model
# ----------------------------
model.save('/content/lstm_autoencoder_model')
print("✅ Model saved successfully at: /content/lstm_autoencoder_model")
