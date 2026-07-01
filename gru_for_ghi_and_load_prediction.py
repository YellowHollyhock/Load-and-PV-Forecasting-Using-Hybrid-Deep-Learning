# -*- coding: utf-8 -*-
"""GRU 
"""

import os, time, tempfile
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from math import sqrt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, GRU
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam

# -------------------------
# Config
# -------------------------
FILE_WEATHER = "/content/drive/MyDrive/data/UTD_Weather.csv"
FILE_PV      = "/content/drive/MyDrive/data/UTD_Load.csv"
TIMESTEPS   = 48
BATCH_SIZE  = 32
EPOCHS      = 100
PATIENCE    = 15
THRESHOLD_DAY = 0.1
TARGET_PV   = "GHI"
TARGET_LOAD = "B4"
PANEL_AREA_M2    = 1.0
PANEL_EFFICIENCY = 0.20

# -------------------------
# Load & preprocess
# -------------------------
weather_df = pd.read_csv(FILE_WEATHER)
load_df    = pd.read_csv(FILE_PV)
weather_df['Timestamp'] = pd.to_datetime(weather_df['Timestamp'])
load_df['Timestamp'] = pd.to_datetime(load_df['Timestamp'])
df = pd.merge(weather_df, load_df, on="Timestamp", how="inner").sort_values("Timestamp").reset_index(drop=True)
df = df.dropna(subset=[TARGET_PV, TARGET_LOAD])
FEATURES = [c for c in df.columns if c not in ["Timestamp", TARGET_PV, TARGET_LOAD]]

# -------------------------
# Helpers
# -------------------------
def create_sequences(X, y, timesteps):
    Xs, ys = [], []
    for i in range(len(X) - timesteps):
        Xs.append(X[i:i+timesteps])
        ys.append(y[i+timesteps])
    return np.array(Xs), np.array(ys)

def safe_mape(y_true, y_pred):
    mask = y_true > THRESHOLD_DAY
    if mask.sum() == 0: return np.nan
    return np.mean(np.abs((y_true[mask]-y_pred[mask])/y_true[mask]))*100

def smape(y_true, y_pred):
    return 100*np.mean(2*np.abs(y_pred-y_true)/(np.abs(y_true)+np.abs(y_pred)+1e-8))

def evaluate(y_true, y_pred, label=""):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    y_range = np.max(y_true)-np.min(y_true)
    nmae = mae/y_range
    nrmse = rmse/y_range
    acc = (1-nmae)*100
    mape = safe_mape(y_true,y_pred)
    smape_val = smape(y_true,smape)
    print(f"\n--- {label} ---")
    print(f"MAE={mae:.3f}, RMSE={rmse:.3f}, R²={r2:.4f}, NMAE={nmae:.4f}, NRMSE={nrmse:.4f}, Acc={acc:.2f}%")
    print(f"MAPE={mape:.2f}%, SMAPE={smape_val:.2f}%")
    return [mae,rmse,r2,nmae,nrmse,acc,mape,smape_val]

def build_gru_model(input_shape):
    model = Sequential([
        GRU(64, return_sequences=True, input_shape=input_shape),
        Dropout(0.2),
        GRU(64, return_sequences=False),
        Dropout(0.2),
        Dense(64, activation="relu"),
        Dense(1, activation="linear")
    ])
    model.compile(optimizer=Adam(0.001), loss="mse")
    return model

def model_size_bytes_and_params(model):
    """Calculates model size in MB and number of parameters."""
    # Calculate model size in bytes
    model_size_bytes = tf.keras.backend.get_value(tf.reduce_sum([tf.reduce_prod(v.shape) for v in model.trainable_variables]) * 4)  # Assuming float32
    model_size_mb = model_size_bytes / (1024 * 1024)

    # Calculate number of parameters
    params = model.count_params()

    return model_size_mb, params

# -------------------------
# Sliding window evaluation
# -------------------------
n = len(df)
window = n // 4  # since N_SPLITS=3, windows = N_SPLITS+1
results = []
last_fold_outputs = {}
for fold in range(3):    # N_SPLITS=3
    print(f"\n--- Fold {fold+1}/3 ---")
    end_train = (fold + 1) * window
    start_test = end_train
    end_test = start_test + window
    train_df = df.iloc[:end_train].reset_index(drop=True)
    test_df = df.iloc[start_test:end_test].reset_index(drop=True)
    if len(test_df) < TIMESTEPS + 2:
        print("Test fold too small, breaking.")
        break

    scaler_X = MinMaxScaler()
    scaler_pv = MinMaxScaler()
    scaler_ld = MinMaxScaler()

    X_train = scaler_X.fit_transform(train_df[FEATURES])
    y_pv_train = scaler_pv.fit_transform(train_df[[TARGET_PV]])
    y_ld_train = scaler_ld.fit_transform(train_df[[TARGET_LOAD]])

    X_test = scaler_X.transform(test_df[FEATURES])
    y_pv_test = scaler_pv.transform(test_df[[TARGET_PV]])
    y_ld_test = scaler_ld.transform(test_df[[TARGET_LOAD]])

    X_tr_seq, y_pv_tr_seq = create_sequences(X_train, y_pv_train, TIMESTEPS)
    _, y_ld_tr_seq = create_sequences(X_train, y_ld_train, TIMESTEPS)
    X_te_seq, y_pv_te_seq = create_sequences(X_test, y_pv_test, TIMESTEPS)
    _, y_ld_te_seq = create_sequences(X_test, y_ld_test, TIMESTEPS)

    model_pv = build_gru_model((TIMESTEPS, X_tr_seq.shape[2]))
    model_ld = build_gru_model((TIMESTEPS, X_tr_seq.shape[2]))
    early = EarlyStopping(monitor='val_loss', patience=PATIENCE, restore_best_weights=True, verbose=0)

    model_pv.fit(X_tr_seq, y_pv_tr_seq, epochs=EPOCHS, batch_size=BATCH_SIZE, validation_split=0.1, verbose=0, callbacks=[early])
    model_ld.fit(X_tr_seq, y_ld_tr_seq, epochs=EPOCHS, batch_size=BATCH_SIZE, validation_split=0.1, verbose=0, callbacks=[early])

    # Latency
    warmup, repeats = 3, 5
    for _ in range(warmup):
        _ = model_pv.predict(X_te_seq, batch_size=BATCH_SIZE, verbose=0)
    t0 = time.time()
    for _ in range(repeats):
        y_pv_pred_scaled = model_pv.predict(X_te_seq, batch_size=BATCH_SIZE, verbose=0)
    t1 = time.time()
    latency_pv = (t1 - t0) / (repeats * len(X_te_seq))
    for _ in range(warmup):
        _ = model_ld.predict(X_te_seq, batch_size=BATCH_SIZE, verbose=0)
    t0 = time.time()
    for _ in range(repeats):
        y_ld_pred_scaled = model_ld.predict(X_te_seq, batch_size=BATCH_SIZE, verbose=0)
    t1 = time.time()
    latency_ld = (t1 - t0) / (repeats * len(X_te_seq))

    y_pv_pred = scaler_pv.inverse_transform(y_pv_pred_scaled).flatten()
    y_pv_true = scaler_pv.inverse_transform(y_pv_te_seq).flatten()
    y_ld_pred = scaler_ld.inverse_transform(y_ld_pred_scaled).flatten()
    y_ld_true = scaler_ld.inverse_transform(y_ld_te_seq).flatten()

    predicted_pv_output_kw = y_pv_pred * PANEL_AREA_M2 * PANEL_EFFICIENCY / 1000.0
    actual_pv_output_kw = y_pv_true * PANEL_AREA_M2 * PANEL_EFFICIENCY / 1000.0
    net_pred = y_ld_pred - predicted_pv_output_kw
    net_true = y_ld_true - actual_pv_output_kw

    def calc_metrics(y_true_arr, y_pred_arr, label):
        mae = mean_absolute_error(y_true_arr, y_pred_arr)
        rmse = sqrt(mean_squared_error(y_true_arr, y_pred_arr))
        r2 = r2_score(y_true_arr, y_pred_arr)
        y_range = (np.max(y_true_arr) - np.min(y_true_arr)) if np.max(y_true_arr) != np.min(y_true_arr) else np.mean(y_true_arr)
        nmae = mae / (y_range + 1e-8)
        nrmse = rmse / (y_range + 1e-8)
        acc = (1 - nmae) * 100.0
        mape_val = safe_mape(y_true_arr, y_pred_arr)
        smape_val = smape(y_true_arr, y_pred_arr)
        return {"label": label, "MAE": mae, "RMSE": rmse, "R2": r2, "NMAE": nmae, "NRMSE": nrmse, "Acc": acc, "MAPE": mape_val, "SMAPE": smape_val}

    metrics_pv = calc_metrics(y_pv_true, y_pv_pred, "GHI")
    metrics_ld = calc_metrics(y_ld_true, y_ld_pred, "Load_B4")
    metrics_net = calc_metrics(net_true, net_pred, "Net_Load")

    size_pv, params_pv = model_size_bytes_and_params(model_pv)
    size_ld, params_ld = model_size_bytes_and_params(model_ld)

    results.append({
        "fold": fold,
        "metrics_g": metrics_pv,
        "metrics_ld": metrics_ld,
        "metrics_net": metrics_net,
        "latency_g_s": latency_pv,
        "latency_ld_s": latency_ld,
        "size_g_mb": size_pv,
        "size_ld_mb": size_ld,
        "params_g": params_pv,
        "params_ld": params_ld
    })

    last_fold_outputs = {
        "y_g_true": y_pv_true, "y_g_pred": y_pv_pred,
        "y_ld_true": y_ld_true, "y_ld_pred": y_ld_pred,
        "pv_true_kw": actual_pv_output_kw, "pv_pred_kw": predicted_pv_output_kw,
        "net_true": net_true, "net_pred": net_pred,
        "timestamps_test": test_df['Timestamp'].iloc[TIMESTEPS:].reset_index(drop=True)
    }

    print(f"Fold {fold+1} metrics (GHI MAE/RMSE/R2): {metrics_pv['MAE']:.3f}/{metrics_pv['RMSE']:.3f}/{metrics_pv['R2']:.4f}")
    print(f"Fold {fold+1} metrics (Load MAE/RMSE/R2): {metrics_ld['MAE']:.3f}/{metrics_ld['RMSE']:.3f}/{metrics_ld['R2']:.4f}")
    print(f"Latencies (ms): GHI {latency_pv*1000:.3f}, Load {latency_ld*1000:.3f}")
    print(f"Model sizes (MB): GHI {size_pv:.2f}, Load {size_ld:.2f}")
    print(f"Params: GHI {params_pv:,}, Load {params_ld:,}")

# Aggregate & print summary if needed (similar to your original)

# -------------------------
# 4. Aggregate & print summary
# -------------------------
def mean_metrics(metrics_list):
    keys = list(metrics_list[0].keys())
    keys.remove('label')
    out = {}
    for k in keys:
        vals = [m[k] for m in metrics_list]
        out[k] = np.nanmean(vals)
    return out

ghi_metrics_list = [r['metrics_g'] for r in results]
ld_metrics_list  = [r['metrics_ld'] for r in results]
net_metrics_list = [r['metrics_net'] for r in results]
ghi_mean = mean_metrics(ghi_metrics_list)
ld_mean  = mean_metrics(ld_metrics_list)
net_mean = mean_metrics(net_metrics_list)

latencies_g = [r['latency_g_s'] for r in results]
latencies_ld = [r['latency_ld_s'] for r in results]
sizes_g = [r['size_g_mb'] for r in results]
sizes_ld = [r['size_ld_mb'] for r in results]
params_g = [r['params_g'] for r in results]
params_ld = [r['params_ld'] for r in results]

print("\n=== Average across folds ===")
print("GHI (avg):", {k: (f"{v:.4f}" if isinstance(v, float) else v) for k,v in ghi_mean.items()})
print("Load (avg):", {k: (f"{v:.4f}" if isinstance(v, float) else v) for k,v in ld_mean.items()})
print("Net  (avg):", {k: (f"{v:.4f}" if isinstance(v, float) else v) for k,v in net_mean.items()})
print(f"Latency (avg ms/sample): GHI {np.mean(latencies_g)*1000:.3f}, Load {np.mean(latencies_ld)*1000:.3f}")
print(f"Model size (avg MB): GHI {np.mean(sizes_g):.2f}, Load {np.mean(sizes_ld):.2f}")
print(f"Params (avg): GHI {int(np.mean(params_g)):,}, Load {int(np.mean(params_ld)):,}")

# -------------------------
# 5. Plots for the last fold
# -------------------------
y_g_true = last_fold_outputs['y_g_true']
y_g_pred = last_fold_outputs['y_g_pred']
y_ld_true = last_fold_outputs['y_ld_true']
y_ld_pred = last_fold_outputs['y_ld_pred']
pv_true_kw = last_fold_outputs['pv_true_kw']
pv_pred_kw = last_fold_outputs['pv_pred_kw']
net_true = last_fold_outputs['net_true']
net_pred = last_fold_outputs['net_pred']
timestamps = last_fold_outputs['timestamps_test']

n_plot = min(500, len(y_g_true))

plt.figure(figsize=(14,4))
plt.plot(timestamps[:n_plot], y_g_true[:n_plot], label="Actual GHI")
plt.plot(timestamps[:n_plot], y_g_pred[:n_plot], label="Predicted GHI", alpha=0.8)
plt.title("Actual vs Predicted GHI (last fold)")
plt.xlabel("Time"); plt.ylabel("GHI (W/m²)")
plt.legend(); plt.grid(True); plt.show()

plt.figure(figsize=(14,4))
plt.plot(timestamps[:n_plot], y_ld_true[:n_plot], label="Actual Load (B4)")
plt.plot(timestamps[:n_plot], y_ld_pred[:n_plot], label="Predicted Load (B4)", alpha=0.8)
plt.title("Actual vs Predicted Load (B4) (last fold)")
plt.xlabel("Time"); plt.ylabel("Load")
plt.legend(); plt.grid(True); plt.show()

plt.figure(figsize=(6,6))
plt.scatter(y_g_true[:n_plot], y_g_pred[:n_plot], s=6, alpha=0.6)
plt.plot([y_g_true[:n_plot].min(), y_g_true[:n_plot].max()],
         [y_g_true[:n_plot].min(), y_g_true[:n_plot].max()], 'r--')
plt.title("Scatter: Actual vs Predicted GHI")
plt.xlabel("Actual GHI"); plt.ylabel("Predicted GHI"); plt.grid(True); plt.show()

plt.figure(figsize=(6,6))
plt.scatter(y_ld_true[:n_plot], y_ld_pred[:n_plot], s=6, alpha=0.6)
plt.plot([y_ld_true[:n_plot].min(), y_ld_true[:n_plot].max()],
         [y_ld_true[:n_plot].min(), y_ld_true[:n_plot].max()], 'r--')
plt.title("Scatter: Actual vs Predicted Load (B4)")
plt.xlabel("Actual Load"); plt.ylabel("Predicted Load"); plt.grid(True); plt.show()

plt.figure(figsize=(14,4))
plt.plot(timestamps[:n_plot], pv_pred_kw[:n_plot], label="Predicted PV (kW)")
plt.plot(timestamps[:n_plot], y_ld_pred[:n_plot], label="Predicted Load (B4)")
plt.title("Predicted PV vs Predicted Load (last fold)")
plt.xlabel("Time"); plt.ylabel("Power (kW or units)"); plt.legend(); plt.grid(True); plt.show()

plt.figure(figsize=(14,4))
plt.plot(timestamps[:n_plot], net_true[:n_plot], label="Actual Net Load")
plt.plot(timestamps[:n_plot], net_pred[:n_plot], label="Predicted Net Load", alpha=0.8)
plt.title("Actual vs Predicted Net Load (last fold)")
plt.xlabel("Time"); plt.ylabel("Net Load"); plt.legend(); plt.grid(True); plt.show()
