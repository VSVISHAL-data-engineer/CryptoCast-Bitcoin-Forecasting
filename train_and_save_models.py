"""
╔══════════════════════════════════════════════════════════════╗
║   TRAIN & SAVE MODELS — Run this ONCE before the Streamlit   ║
║   app. It trains all 12 models (4 architectures × 3 horizons)║
║   and saves them + the scalers to the "saved_models" folder. ║
╚══════════════════════════════════════════════════════════════╝
"""

import pandas as pd
import numpy as np
import pickle
import os
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import warnings
warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

torch.manual_seed(42)
np.random.seed(42)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

os.makedirs('saved_models', exist_ok=True)
print(f"✅ Device: {DEVICE}")


# ─────────────────────────────────────────────
# LOAD & PREPROCESS DATA
# ─────────────────────────────────────────────
CSV_PATH = "Bitcoin Historical Data (1) (2).csv"   # ⚠️ change if your filename differs

df = pd.read_csv(CSV_PATH)

def preprocess(df):
    data = df.copy()
    data['Date'] = pd.to_datetime(data['Date'], dayfirst=True)
    data = data.sort_values('Date').reset_index(drop=True)

    for col in ['Price', 'Open', 'High', 'Low']:
        data[col] = data[col].astype(str).str.replace(',', '').astype(float)

    def fix_vol(v):
        if isinstance(v, str):
            v = v.strip()
            if 'K' in v: return float(v.replace('K','')) * 1e3
            if 'M' in v: return float(v.replace('M','')) * 1e6
            if 'B' in v: return float(v.replace('B','')) * 1e9
            try: return float(v)
            except: return np.nan
        return v

    data['Vol.'] = data['Vol.'].apply(fix_vol)
    data['Change %'] = data['Change %'].astype(str).str.replace('%','').astype(float)
    data = data.ffill().bfill()
    return data

data = preprocess(df)
print(f"✅ Data loaded | {data['Date'].min().date()} → {data['Date'].max().date()} | Rows: {len(data)}")

# Save the cleaned data for the Streamlit app to display
data.to_csv('saved_models/cleaned_data.csv', index=False)


# ─────────────────────────────────────────────
# SCALE & SEQUENCE
# ─────────────────────────────────────────────
FEATURES = ['Price', 'Open', 'High', 'Low', 'Vol.', 'Change %']
SEQ_LEN  = 60
HORIZONS = [1, 3, 7]

scaler       = MinMaxScaler()
price_scaler = MinMaxScaler()
scaled = scaler.fit_transform(data[FEATURES])
price_scaler.fit(data[['Price']])

# Save scalers — the Streamlit app needs these to scale new input & inverse-transform predictions
with open('saved_models/scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)
with open('saved_models/price_scaler.pkl', 'wb') as f:
    pickle.dump(price_scaler, f)

def make_sequences(scaled, seq_len, horizon, price_idx=0):
    X, y = [], []
    for i in range(seq_len, len(scaled) - horizon + 1):
        X.append(scaled[i-seq_len:i])
        y.append(scaled[i+horizon-1, price_idx])
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)

def time_split(X, y, ratio=0.2):
    s = int(len(X)*(1-ratio))
    return X[:s], X[s:], y[:s], y[s:]

datasets = {}
for h in HORIZONS:
    X, y = make_sequences(scaled, SEQ_LEN, h)
    Xtr, Xte, ytr, yte = time_split(X, y)
    datasets[h] = dict(Xtr=Xtr, Xte=Xte, ytr=ytr, yte=yte)

N_FEAT = scaled.shape[1]
print(f"✅ Sequences ready | Input: ({SEQ_LEN}, {N_FEAT})")


# ─────────────────────────────────────────────
# MODEL DEFINITIONS (must match the app's definitions exactly)
# ─────────────────────────────────────────────
class CNNModel(nn.Module):
    def __init__(self, seq_len, n_feat):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(n_feat, 64, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=3, padding=1),    nn.ReLU(),
            nn.AdaptiveAvgPool1d(8),
            nn.Flatten(),
            nn.Linear(128*8, 64), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(64, 1)
        )
    def forward(self, x):
        return self.net(x.permute(0,2,1)).squeeze(-1)


class RNNModel(nn.Module):
    def __init__(self, seq_len, n_feat):
        super().__init__()
        self.rnn = nn.RNN(n_feat, 64, num_layers=2, batch_first=True, dropout=0.2)
        self.fc  = nn.Sequential(nn.Linear(64,32), nn.ReLU(), nn.Linear(32,1))
    def forward(self, x):
        out, _ = self.rnn(x)
        return self.fc(out[:,-1,:]).squeeze(-1)


class LSTMModel(nn.Module):
    def __init__(self, seq_len, n_feat):
        super().__init__()
        self.lstm = nn.LSTM(n_feat, 64, num_layers=2, batch_first=True, dropout=0.2)
        self.fc   = nn.Sequential(nn.Linear(64,32), nn.ReLU(), nn.Dropout(0.2), nn.Linear(32,1))
    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:,-1,:]).squeeze(-1)


class TransformerModel(nn.Module):
    def __init__(self, seq_len, n_feat):
        super().__init__()
        self.input_proj = nn.Linear(n_feat, 64)
        encoder_layer   = nn.TransformerEncoderLayer(d_model=64, nhead=4,
                            dim_feedforward=128, dropout=0.1, batch_first=True)
        self.encoder    = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.fc         = nn.Sequential(nn.Linear(64,32), nn.ReLU(), nn.Dropout(0.2), nn.Linear(32,1))
    def forward(self, x):
        x = self.input_proj(x)
        x = self.encoder(x)
        return self.fc(x.mean(dim=1)).squeeze(-1)


MODEL_CLASSES = {'1D-CNN': CNNModel, 'RNN': RNNModel,
                 'LSTM': LSTMModel, 'Transformer': TransformerModel}


# ─────────────────────────────────────────────
# TRAIN FUNCTION
# ─────────────────────────────────────────────
def train_model(model, Xtr, ytr, epochs=60, batch=32, lr=1e-3, patience=10):
    model.to(DEVICE)
    opt       = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    val_split = int(len(Xtr)*0.85)
    Xv = torch.tensor(Xtr[val_split:]).to(DEVICE)
    yv = torch.tensor(ytr[val_split:]).to(DEVICE)
    Xt = torch.tensor(Xtr[:val_split]).to(DEVICE)
    yt = torch.tensor(ytr[:val_split]).to(DEVICE)

    best_val, patience_cnt, best_state = 999, 0, None

    for ep in range(epochs):
        model.train()
        for xb, yb in DataLoader(TensorDataset(Xt, yt), batch_size=batch, shuffle=False):
            opt.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            opt.step()

        model.eval()
        with torch.no_grad():
            vl = criterion(model(Xv), yv).item()

        if vl < best_val:
            best_val, patience_cnt = vl, 0
            best_state = {k:v.clone() for k,v in model.state_dict().items()}
        else:
            patience_cnt += 1
            if patience_cnt >= patience:
                break

    model.load_state_dict(best_state)
    return model


# ─────────────────────────────────────────────
# TRAIN ALL 12 MODELS, SAVE EACH, COLLECT METRICS
# ─────────────────────────────────────────────
results_summary = []

print("\n🚀 Training & saving all models...\n")

for mname, MClass in MODEL_CLASSES.items():
    for h in HORIZONS:
        print(f"  ⏳ {mname} | {h}D ...", end=' ', flush=True)
        d = datasets[h]

        model = MClass(SEQ_LEN, N_FEAT)
        model = train_model(model, d['Xtr'], d['ytr'])

        model.eval()
        with torch.no_grad():
            Xte_t = torch.tensor(d['Xte']).to(DEVICE)
            pred  = model(Xte_t).cpu().numpy()

        y_actual = price_scaler.inverse_transform(d['yte'].reshape(-1,1)).flatten()
        y_pred   = price_scaler.inverse_transform(pred.reshape(-1,1)).flatten()

        mae  = mean_absolute_error(y_actual, y_pred)
        rmse = np.sqrt(mean_squared_error(y_actual, y_pred))
        mape = np.mean(np.abs((y_actual - y_pred)/y_actual)) * 100

        # Save model weights
        fname = f"saved_models/{mname.replace('1D-','')}_{h}D.pt"
        torch.save(model.state_dict(), fname)

        results_summary.append(dict(model=mname, horizon=h, mae=mae, rmse=rmse, mape=mape))
        print(f"MAE=${mae:,.0f} | MAPE={mape:.2f}% → saved {fname}")

# Save metrics table for the app to show
results_df = pd.DataFrame(results_summary)
results_df.to_csv('saved_models/results_summary.csv', index=False)

# Save config (seq_len, features, horizons) so the app knows how to rebuild models
config = dict(SEQ_LEN=SEQ_LEN, FEATURES=FEATURES, HORIZONS=HORIZONS, N_FEAT=N_FEAT)
with open('saved_models/config.pkl', 'wb') as f:
    pickle.dump(config, f)

print("\n✅ All models trained and saved in 'saved_models/' folder!")
print("✅ Now run: streamlit run app.py")
